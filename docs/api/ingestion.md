# Ingestion API Reference

## Protocol

All loader backends implement the `DocumentLoader` Protocol from `ingestion/base.py`.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `(source: str) → list[dict]` | Load and chunk documents from source path |

### Output Schema

Each element in the returned list:
```python
{
    "text": str,       # chunk text content
    "metadata": {
        "source": str,   # file path or identifier
        # loader-specific fields below
    }
}
```

## Backends

### PDF Loader (`loader: "pdf"`)

- **Install:** `pip install pypdf`
- **Source:** Path to `.pdf` file
- **Metadata fields:** `source`, `page` (0-indexed page number), `section`, `chunk_id`
- **Config fields:** `chunk_size`, `chunk_overlap`

### Markdown Loader (`loader: "markdown"`)

- **Source:** Path to `.md` file
- **Splits on:** ATX headings (`#`, `##`, `###`)
- **Metadata fields:** `source`, `heading`, `section`, `chunk_id`

### JSONL Loader (`loader: "jsonl"`)

- **Source:** Path to `.jsonl` file (one JSON object per line)
- **Config fields:** `text_key` (which JSON field contains the text, scoped to config)
- **Metadata fields:** `source`, `line`, `chunk_id`

### HTML Loader (`loader: "html"`)

- **Install:** `pip install beautifulsoup4`
- **Source:** Path to `.html` file
- **Strips:** All HTML tags; extracts visible text only
- **Metadata fields:** `source`, `section`, `chunk_id`

### Directory Loader (`loader: "directory"`)

- **Source:** Path to a directory
- **Dispatches by extension:** `.pdf` → PDFLoader, `.md` → MarkdownLoader, `.jsonl` → JSONLLoader, `.html` → HTMLLoader
- **Skips:** Unknown file types
- **Metadata fields:** per-loader fields + `source` (full path)

## Output consistency

All loaders:
- Skip chunks with fewer than `min_chunk_words` words (default: 5)
- Include `"text"` and `"metadata"` keys in every element
- Include `"source"` in every `"metadata"` dict
