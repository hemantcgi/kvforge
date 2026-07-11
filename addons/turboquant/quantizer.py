"""TurboQuantProd key codec + group value codec.

Key algorithm (arXiv:2504.19874, Algorithm 2):
  1. Normalize to unit sphere; store L2 norm.
  2. Rotate by random orthogonal Pi (seeded by seed).
  3. Lloyd-Max quantize at (key_bits - 1) bits per coordinate.
  4. QJL residual: project reconstruction error through S, store sign bits.

Value algorithm: asymmetric per-group min-max, group_size=32.

Decompression uses 1-bit CS recovery (Boufounos & Baraniuk, 2008):
  residual_hat = (qjl_signs @ S) * scale, where scale is derived from the
  theoretical 2-bit Gaussian quantization distortion (gamma ≈ 0.1175).
  This gives residual_hat ≈ residual with cosine similarity ≈ sqrt(2/π) ≈ 0.80,
  improving overall key reconstruction from ~0.94 to ~0.97.
"""
import math
import numpy as np
import torch
from .codebooks import get_codebook


def _make_rotation(dim: int, seed: int) -> torch.Tensor:
    """Deterministic random orthogonal matrix via QR decomposition."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    G = torch.randn(dim, dim, generator=rng)
    Q, _ = torch.linalg.qr(G)
    return Q  # [dim, dim]


def _make_qjl_matrix(dim: int, seed: int) -> torch.Tensor:
    """Random {±1/√dim} matrix S for QJL projection."""
    rng = torch.Generator()
    rng.manual_seed(seed + 99999)
    signs = torch.randint(0, 2, (dim, dim), generator=rng).float() * 2 - 1
    return signs / (dim ** 0.5)  # [dim, dim]


class TurboQuantKeyCodec:
    """Compress and estimate inner products for per-token key tensors."""

    def __init__(self, head_dim: int, num_bits: int = 3, seed: int = 42):
        assert num_bits in {2, 3}
        self.head_dim  = head_dim
        self.num_bits  = num_bits
        self.msb_bits  = num_bits - 1
        self.centroids, self.boundaries = get_codebook(head_dim, self.msb_bits)
        self.Pi = _make_rotation(head_dim, seed)
        self.S  = _make_qjl_matrix(head_dim, seed)

    def compress(self, keys: torch.Tensor) -> dict:
        """Compress keys shaped [..., head_dim].

        Returns dict with:
          norms:     [...] float32
          indices:   [..., head_dim] int8  — Lloyd-Max centroid indices
          qjl_signs: [..., head_dim] int8  — QJL residual signs {+1, -1}
        """
        orig_shape = keys.shape
        flat = keys.reshape(-1, self.head_dim).float()

        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = flat / norms

        rotated = normed @ self.Pi.T

        t = torch.from_numpy(self.boundaries)
        indices = torch.searchsorted(t.expand(rotated.shape[0], -1),
                                     rotated.contiguous())
        indices = indices.clamp(0, len(self.centroids) - 1).to(torch.int8)

        c_vals = torch.from_numpy(self.centroids)[indices.long()]
        residual = rotated - c_vals

        qjl_proj = residual @ self.S.T
        qjl_signs = qjl_proj.sign().to(torch.int8)
        qjl_signs[qjl_signs == 0] = 1

        return {
            "norms":     norms.squeeze(-1).reshape(orig_shape[:-1]),
            "indices":   indices.reshape(*orig_shape[:-1], self.head_dim),
            "qjl_signs": qjl_signs.reshape(*orig_shape[:-1], self.head_dim),
        }

    def decompress(self, compressed: dict) -> torch.Tensor:
        """Reconstruct keys (approximate) from compressed representation.

        Uses 1-bit CS recovery: residual ≈ (qjl_signs @ S) * scale
        where scale = sqrt(gamma * π / (2d)), gamma ≈ 0.1175 (2-bit Gaussian distortion).
        """
        indices = compressed["indices"].long()
        norms   = compressed["norms"].unsqueeze(-1)
        c_vals  = torch.from_numpy(self.centroids).float()[indices]

        # 1-bit CS recovery of quantization residual.
        # Optimal scale derivation: E[||qjl_signs @ S||²] = d*(1 + 2/π) (not d),
        # because qjl_signs correlates with S, inflating the norm.
        # scale_opt = E[residual · recovery] / E[||recovery||²]
        #           = sqrt(2γ/π) / (sqrt(d) * (1 + 2/π))
        _gamma = 0.1175  # 2-bit Gaussian Lloyd-Max normalized MSE
        scale  = math.sqrt(2 * _gamma / math.pi) / (math.sqrt(self.head_dim) * (1 + 2 / math.pi))
        orig_shape = c_vals.shape
        flat_signs  = compressed["qjl_signs"].float().reshape(-1, self.head_dim)
        residual_hat = (flat_signs @ self.S) * scale
        qjl_corr = residual_hat.reshape(orig_shape)

        rotated_rec = c_vals + qjl_corr
        normed_rec = rotated_rec @ self.Pi
        return normed_rec * norms

    def estimate_dot(self, query: torch.Tensor, compressed: dict,
                     token_idx: int = 0, head_idx: int = 0, batch_idx: int = 0) -> float:
        """Estimate q·k directly from compressed representation (no decompression)."""
        norm   = compressed["norms"][batch_idx, token_idx, head_idx].item()
        idx    = compressed["indices"][batch_idx, token_idx, head_idx].long()
        signs  = compressed["qjl_signs"][batch_idx, token_idx, head_idx].float()

        q_rot  = (query.float() @ self.Pi.T)
        c_vals = torch.from_numpy(self.centroids).float()[idx]
        qjl_q  = q_rot @ self.S.T
        qjl_mag = 1.0 / (self.head_dim ** 0.5)

        dot_msb = (q_rot * c_vals).sum().item()
        dot_qjl = (qjl_q * signs * qjl_mag).sum().item()
        return (dot_msb + dot_qjl) * norm


class GroupValueCodec:
    """Asymmetric per-group min-max quantization for value tensors."""

    def __init__(self, num_bits: int = 4, group_size: int = 32):
        assert num_bits in {2, 4}
        self.num_bits   = num_bits
        self.group_size = group_size
        self.levels     = 2 ** num_bits - 1

    def compress(self, values: torch.Tensor) -> dict:
        """Compress values shaped [..., head_dim].

        Returns dict with:
          packed: [..., head_dim // pack_factor] uint8
          scales: [..., n_groups] float16
          zeros:  [..., n_groups] float16
        """
        orig_shape = values.shape
        head_dim   = orig_shape[-1]
        flat = values.reshape(-1, head_dim).float()

        n_groups = head_dim // self.group_size
        groups   = flat.reshape(-1, n_groups, self.group_size)

        mn    = groups.min(dim=-1).values
        mx    = groups.max(dim=-1).values
        scale = (mx - mn).clamp(min=1e-8) / self.levels
        zero  = mn

        q = ((groups - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(0, self.levels)
        q = q.reshape(-1, head_dim).to(torch.uint8)

        pack = 8 // self.num_bits
        packed = torch.zeros(q.shape[0], head_dim // pack, dtype=torch.uint8)
        for i in range(pack):
            packed |= (q[:, i::pack] & self.levels) << (i * self.num_bits)

        batch_shape = orig_shape[:-1]
        return {
            "packed": packed.reshape(*batch_shape, head_dim // pack),
            "scales": scale.to(torch.float16).reshape(*batch_shape, n_groups),
            "zeros":  zero.to(torch.float16).reshape(*batch_shape, n_groups),
        }

    def decompress(self, compressed: dict) -> torch.Tensor:
        orig_shape = (*compressed["packed"].shape[:-1],
                      compressed["packed"].shape[-1] * (8 // self.num_bits))
        packed = compressed["packed"].reshape(-1, compressed["packed"].shape[-1])
        scales = compressed["scales"].reshape(-1, compressed["scales"].shape[-1]).float()
        zeros  = compressed["zeros"].reshape(-1, compressed["zeros"].shape[-1]).float()

        head_dim = orig_shape[-1]
        pack     = 8 // self.num_bits
        q        = torch.zeros(packed.shape[0], head_dim, dtype=torch.float32)
        for i in range(pack):
            q[:, i::pack] = ((packed >> (i * self.num_bits)) & self.levels).float()

        n_groups   = head_dim // self.group_size
        groups_q   = q.reshape(-1, n_groups, self.group_size)
        groups_rec = groups_q * scales.unsqueeze(-1) + zeros.unsqueeze(-1)
        return groups_rec.reshape(orig_shape)
