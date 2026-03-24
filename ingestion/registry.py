"""ingestion/registry.py — Factory for DocumentLoader implementations."""


def get_loader(cfg: dict):
    """Return the appropriate DocumentLoader for the given config.

    Dispatches on cfg['loader'] (default: 'pdf').
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
    raise ValueError(
        f"Unknown loader '{name}'. Choose: pdf, markdown, jsonl, html, directory"
    )
