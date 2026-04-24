"""ImageLoader protocol and PDFImageExtractor implementation.

Extracts raster images from PDF pages using pdfplumber, saves each as a PNG,
and returns structured dicts for downstream embedding and KV computation.
"""
import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import pdfplumber


_MIN_IMAGE_PX = 32   # images smaller than this in either dimension are skipped


@runtime_checkable
class ImageLoader(Protocol):
    def load(self, source: str) -> list[dict]: ...
    # Returns: [{"image_path": str, "page": int, "source_file": str}, ...]


class PDFImageExtractor:
    """Extracts raster images from PDF files.

    Each extracted image is saved as a PNG file in
    ``<image_store_dir>/<collection>/`` and represented as a dict with keys
    ``image_path``, ``page`` (1-indexed), and ``source_file``.
    """

    def __init__(self, cfg: dict) -> None:
        image_store_dir = cfg.get("image_store_dir", "")
        if not image_store_dir:
            raise ValueError(
                "image_store_dir must be set in config to use PDFImageExtractor"
            )
        self._out_dir = Path(image_store_dir) / cfg.get("collection", "default")
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def load(self, source: str) -> list[dict]:
        source_path = Path(source)
        source_stem = source_path.stem
        source_file = source_path.name
        results = []

        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                page_num = page.page_number  # 1-indexed
                for idx, img_obj in enumerate(page.images):
                    w = img_obj.get("width", 0)
                    h = img_obj.get("height", 0)
                    if w < _MIN_IMAGE_PX or h < _MIN_IMAGE_PX:
                        continue

                    pil_image = page.to_image(resolution=150).original
                    fname = f"{source_stem}_p{page_num}_{idx}.png"
                    out_path = self._out_dir / fname
                    pil_image.save(str(out_path), format="PNG")

                    results.append({
                        "image_path": str(out_path),
                        "page": page_num,
                        "source_file": source_file,
                    })

        return results
