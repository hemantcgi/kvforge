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
    by the appropriate loader. Results are concatenated in alphabetical order
    by file path.

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
            in the archive, in alphabetical order by file path.
        """
        docs = []

        # Extract ZIP to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(source, "r") as zf:
                # Selectively extract only supported files before extracting
                tmppath = Path(tmpdir)
                tmppath_resolved = tmppath.resolve()

                for member in zf.namelist():
                    # Skip directories
                    if member.endswith("/"):
                        continue
                    # Skip nested ZIPs
                    if member.lower().endswith(".zip"):
                        continue
                    # Only extract supported formats
                    suffix = Path(member).suffix.lower()
                    if suffix not in EXTENSION_TO_LOADER:
                        continue
                    # Guard against path traversal (normalize and check it stays within tmpdir)
                    member_path = (tmppath / member).resolve()
                    if not str(member_path).startswith(str(tmppath_resolved)):
                        continue  # skip path traversal attempts
                    zf.extract(member, tmpdir)

            # Walk extracted files in sorted order
            for file_path in sorted(tmppath.glob("**/*")):
                if not file_path.is_file():
                    continue

                ext = file_path.suffix.lower()

                # Skip unsupported extensions (double-check, though we pre-filtered)
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
            return PptxLoader(**self.loader_kwargs)
        elif loader_class_name == "XlsxLoader":
            from ingestion.xlsx_loader import XlsxLoader
            return XlsxLoader(**self.loader_kwargs)
        else:
            raise ValueError(f"Unknown loader: {loader_class_name}")
