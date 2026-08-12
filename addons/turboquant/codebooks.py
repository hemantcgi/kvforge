"""Lloyd-Max codebooks for the Gaussian distribution of rotated unit-sphere coordinates.

After applying a random orthogonal matrix to a unit-norm key vector in R^d, each
coordinate follows approximately N(0, 1/d) for large d (the arcsine distribution
concentrates near 0 as d grows). Codebooks are therefore head_dim-dependent.

Centroids are Gaussian Lloyd-Max optimal values scaled by σ = 1/sqrt(head_dim):
  2-bit multiples of σ: {-1.5098, -0.4528, 0.4528, 1.5098}
  3-bit multiples of σ: {-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520}
"""
import math
import numpy as np
from functools import lru_cache

# Gaussian Lloyd-Max centroids normalized by σ (dimension-independent part)
_SIGMA_MULTIPLES = {
    2: np.array([-1.5098, -0.4528,  0.4528,  1.5098], dtype=np.float32),
    3: np.array([-2.1520, -1.3439, -0.7560, -0.2451,
                  0.2451,  0.7560,  1.3439,  2.1520], dtype=np.float32),
}


@lru_cache(maxsize=16)
def get_codebook(head_dim: int, num_bits: int):
    """Return (centroids, boundaries) for Lloyd-Max quantization.

    Both are 1-D numpy float32 arrays.
    centroids:  2^num_bits values scaled for N(0, 1/head_dim)
    boundaries: 2^num_bits - 1 decision boundaries (midpoints between centroids)
    """
    assert head_dim in {64, 96, 128, 256}, f"Unsupported head_dim {head_dim}"
    assert num_bits in {2, 3}, f"Unsupported num_bits {num_bits}"
    sigma = 1.0 / math.sqrt(head_dim)
    centroids  = (_SIGMA_MULTIPLES[num_bits] * sigma).copy()
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries
