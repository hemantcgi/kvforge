"""Factory that instantiates the configured DocumentLoader backend.

Provides a single ``get_loader`` entry-point used by indexing pipelines so
that the rest of the codebase stays format-agnostic.
"""


def get_loader(cfg: dict):
    """Return the appropriate DocumentLoader for the given config.

    Dispatches on ``cfg['loader']`` (default: ``'pdf'``).  Additional keys
    in *cfg* are forwarded to the loader constructor where applicable (e.g.
    ``chunk_size``, ``chunk_overlap``, ``jsonl_text_key``).

    Args:
        cfg: Datasource configuration dictionary.  Relevant keys:

            * ``loader`` — one of ``pdf``, ``markdown``, ``jsonl``, ``html``,
              ``directory``, ``docx``, ``pptx``, ``xlsx``, ``zip``.
            * ``chunk_size`` — word-count target per chunk (default 600).
            * ``chunk_overlap`` — word overlap between consecutive chunks
              (default 60).
            * ``jsonl_text_key`` — field name for the text in each JSONL object
              (default ``'text'``).
            * ``rows_per_chunk`` — rows per chunk for xlsx loader (default 50).

    Returns:
        A ``DocumentLoader``-protocol-compatible instance.

    Raises:
        ValueError: If ``cfg['loader']`` is not a recognised loader name.
    """
    name = cfg.get("loader", "pdf")
    chunk_size = cfg.get("chunk_size", 600)
    chunk_overlap = cfg.get("chunk_overlap", 60)

    if name == "pdf":
        from ingestion.pdf_loader import PDFLoader
        return PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if name == "markdown":
        from ingestion.markdown_loader import MarkdownLoader
        return MarkdownLoader()
    if name == "jsonl":
        text_key = cfg.get("jsonl_text_key", "text")
        from ingestion.jsonl_loader import JSONLLoader
        return JSONLLoader(text_key=text_key)
    if name == "html":
        from ingestion.html_loader import HTMLLoader
        return HTMLLoader()
    if name == "directory":
        from ingestion.directory_loader import DirectoryLoader
        return DirectoryLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if name == "docx":
        from ingestion.docx_loader import DocxLoader
        return DocxLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if name == "pptx":
        from ingestion.pptx_loader import PptxLoader
        return PptxLoader()
    if name == "xlsx":
        from ingestion.xlsx_loader import XlsxLoader
        kwargs = {}
        if "rows_per_chunk" in cfg:
            kwargs["rows_per_chunk"] = cfg["rows_per_chunk"]
        return XlsxLoader(**kwargs)
    if name == "zip":
        from ingestion.zip_loader import ZipLoader
        return ZipLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if name == "huggingface":
        from ingestion.huggingface_loader import HuggingFaceLoader
        return HuggingFaceLoader(
            dataset_id=cfg.get("dataset_id", ""),
            config_name=cfg.get("hf_config_name"),
            split=cfg.get("hf_split", "train"),
            text_column=cfg.get("hf_text_column", "text"),
            max_rows=int(cfg.get("hf_max_rows", 0)),
            hf_token=cfg.get("hf_token"),
            trust_remote_code=bool(cfg.get("hf_trust_remote_code", False)),
        )
    raise ValueError(
        f"Unknown loader '{name}'. Choose: pdf, markdown, jsonl, html, directory, docx, pptx, xlsx, zip, huggingface"
    )
