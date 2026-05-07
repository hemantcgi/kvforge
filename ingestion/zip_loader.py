"""ZIP archive loader that dispatches to per-format loaders.

Extracts ZIP contents to a temporary directory and dispatches each file
to the appropriate per-format loader based on extension.

Supported extensions: ``.docx``, ``.pptx``, ``.xlsx``.
Nested ZIP files are silently skipped (not recursed).
Unsupported extensions are silently skipped.
"""
import tempfile
import zipfile
from pathlib import Path


# Map file extensions to loader classes (not loader names from registry)
EXTENSION_TO_LOADER = {
    ".docx": "DocxLoader",
    ".pptx": "PptxLoader",
    ".xlsx": "XlsxLoader",
}


class ZipLoader:
    """Load a ZIP archive and dispatch files to per-format loaders.

    Each supported file (based on extension) is extracted and processed
    by the appropriate loader. Results are concatenated in the order files
    appear in the ZIP archive.

    Args:
        **loader_kwargs: Additional keyword arguments forwarded to each
            per-format loader (e.g. ``chunk_size``, ``chunk_overlap``,
            ``rows_per_chunk``).
    """

    def __init__(self, **loader_kwargs):
        self.loader_kwargs = loader_kwargs

    def load(self, source: str) -> list[dict]:
        """Load all supported documents from a ZIP archive.

        Args:
            source: Path to the ZIP file.

        Returns:
            Concatenated list of document dicts from all supported files
            in the archive, in the order they appear.
        """
        docs = []

        # Extract ZIP to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(source, "r") as zf:
                # Extract all files
                zf.extractall(tmpdir)

            # Walk extracted files in sorted order
            tmppath = Path(tmpdir)
            for file_path in sorted(tmppath.glob("**/*")):
                if not file_path.is_file():
                    continue

                ext = file_path.suffix.lower()

                # Skip nested ZIP files
                if ext == ".zip":
                    continue

                # Skip unsupported extensions
                if ext not in EXTENSION_TO_LOADER:
                    continue

                # Dispatch to the appropriate loader
                loader_class_name = EXTENSION_TO_LOADER[ext]
                loader = self._get_loader_instance(loader_class_name)

                docs.extend(loader.load(str(file_path)))

        return docs

    def _get_loader_instance(self, loader_class_name: str):
        """Instantiate a loader by class name.

        Args:
            loader_class_name: Name of the loader class (e.g., 'DocxLoader').

        Returns:
            An instantiated loader.
        """
        if loader_class_name == "DocxLoader":
            from ingestion.docx_loader import DocxLoader
            return DocxLoader(**self.loader_kwargs)
        elif loader_class_name == "PptxLoader":
            from ingestion.pptx_loader import PptxLoader
            return PptxLoader()
        elif loader_class_name == "XlsxLoader":
            from ingestion.xlsx_loader import XlsxLoader
            return XlsxLoader(**self.loader_kwargs)
        else:
            raise ValueError(f"Unknown loader: {loader_class_name}")
