# Document Ingestion

← [Back to FAQ index](../../FAQ.md)

---

### How do I index Markdown documentation?

#### Setup

```bash
# No extra dependencies needed for Markdown
python smartqdrant.py init --name docs --loader markdown
```

#### Index a single file

```bash
python smartqdrant.py index \
  --config datasource_docs.json \
  --source ./README.md
```

#### Index a directory of Markdown files

Change the loader to `directory` so mixed `.md`/`.pdf`/`.html` files are handled automatically:

```bash
python smartqdrant.py init --name docs --loader directory
python smartqdrant.py index \
  --config datasource_docs.json \
  --source ./docs/
```

#### How the Markdown loader splits content

The loader splits on `#`, `##`, and `###` headings using a regex split. Each heading becomes a new chunk. The heading text itself is preserved as the first line of the chunk body. Sections with fewer than 10 words (the `min_chunk_words` threshold) are silently skipped.

Example — given this file:

```markdown
# Installation

Run `pip install smartqdrant` to install.

## Configuration

Copy `datasource_template.json` and edit the fields.

## Troubleshooting

Check logs in `kv_background.log`.
```

The loader produces three chunks: one per `##` section.

#### Tuning for long documents

If your Markdown files contain very long sections without headings, consider switching to `loader: directory` and adding a post-processing step to further split large chunks. Alternatively, pre-process your Markdown files to add intermediate `##` headings.

---

### How do I index a JSONL dataset?

JSONL (JSON Lines) is common for datasets, evaluation sets, and structured knowledge bases.

#### Setup

No extra dependencies required.

```bash
python smartqdrant.py init --name knowledge-base --loader jsonl
```

#### Config

```json
{
  "loader":         "jsonl",
  "jsonl_text_key": "content"
}
```

`jsonl_text_key` (default `"text"`) names the field SmartQdrant reads as the chunk text. All other fields in each JSON object are stored in metadata and available in search results via `payload`.

#### Example JSONL formats

Standard format (default config, `jsonl_text_key: "text"`):
```jsonl
{"text": "Retrieval-Augmented Generation combines dense retrieval with generative models.", "source": "paper_1", "year": 2020}
{"text": "LoRA enables parameter-efficient fine-tuning by decomposing weight updates.", "source": "paper_2", "year": 2022}
```

Custom field name (`jsonl_text_key: "content"`):
```jsonl
{"id": "doc-001", "content": "SmartQdrant stores KV tensors in Qdrant payload fields.", "category": "architecture"}
{"id": "doc-002", "content": "Phase 2 activates after PRS exceeds the configured threshold.", "category": "phases"}
```

HuggingFace datasets export format (`jsonl_text_key: "passage"`):
```jsonl
{"passage": "The Earth is approximately 4.5 billion years old.", "title": "Earth", "source": "wiki"}
```

#### Indexing

```bash
python smartqdrant.py index \
  --config datasource_knowledge-base.json \
  --source ./my_dataset.jsonl
```

#### Accessing metadata in search results

```python
results = store.query("knowledge-base", query_vec, top_k=5)
for r in results:
    print(r.score, r.payload["source"], r.payload["text"][:100])
```

---

### How do I index HTML pages or web content?

#### Setup

```bash
pip install beautifulsoup4
python smartqdrant.py init --name web-corpus --loader html
```

#### Index a single HTML file

```bash
python smartqdrant.py index \
  --config datasource_web-corpus.json \
  --source ./pages/article.html
```

#### How the HTML loader works

1. Reads the file with UTF-8 encoding
2. Parses with `BeautifulSoup(html, "html.parser")`
3. Extracts all visible text with `soup.get_text(separator=" ", strip=True)`
4. Splits the cleaned text into overlapping word-level chunks (using `chunk_size` and `chunk_overlap` from config)
5. Skips chunks with fewer than `min_chunk_words` (default 10) words

Script tags, style tags, and all HTML markup are stripped. Only visible text content is kept.

#### Downloading pages from the web before indexing

SmartQdrant does not crawl URLs directly. Download HTML first:

```bash
# Single page
curl -L https://example.com/docs/page > ./pages/page.html

# Multiple pages with wget
wget -r -l 2 -A .html -P ./pages/ https://example.com/docs/

# Python — download a list of URLs
python - <<'EOF'
import httpx
from pathlib import Path

urls = [
    "https://example.com/docs/intro",
    "https://example.com/docs/api",
]
Path("pages").mkdir(exist_ok=True)
for i, url in enumerate(urls):
    resp = httpx.get(url, follow_redirects=True)
    Path(f"pages/page_{i}.html").write_bytes(resp.content)
    print(f"Downloaded {url}")
EOF
```

Then index the downloaded directory:

```bash
python smartqdrant.py init --name web-corpus --loader directory
python smartqdrant.py index \
  --config datasource_web-corpus.json \
  --source ./pages/
```

---

### How do I index an entire directory of mixed file types?

```bash
python smartqdrant.py init --name mixed-corpus --loader directory
python smartqdrant.py index \
  --config datasource_mixed-corpus.json \
  --source ./corpus/
```

The directory loader walks the directory recursively and dispatches each file by extension:

| Extension | Loader used | Extra dependency |
|-----------|-------------|:----------------:|
| `.pdf` | PDFLoader | `pypdf` |
| `.md`, `.markdown` | MarkdownLoader | none |
| `.jsonl` | JSONLLoader | none |
| `.html`, `.htm` | HTMLLoader | `beautifulsoup4` |
| anything else | Skipped | — |

Files that are skipped are logged to stdout. If you need to index `.txt` or `.rst` files, add a plain-text loader (see [How do I add support for a custom document format?](#how-do-i-add-support-for-a-custom-document-format)).

#### Chunk sizing across formats

All loaders respect the same `chunk_size` and `chunk_overlap` config values. The PDF and HTML loaders use word-level chunking. The Markdown loader splits on headings rather than word count. The JSONL loader treats each JSON object as one chunk (no further splitting).

If you are mixing PDFs (which need word-level splitting) with Markdown (which splits by heading), be aware that Markdown chunks may be smaller than `chunk_size`. This is usually fine — the model handles variable-length chunks well.

---

### How do I add support for a custom document format?

#### Step 1 — Implement the DocumentLoader protocol

The protocol is in `ingestion/base.py` and requires a single method: `load(source: str) -> list[dict]`. Each returned dict must have a `"text"` key (string) and a `"metadata"` key (dict containing at least `"chunk_id"` and `"source"`).

```python
# ingestion/csv_loader.py
import csv
from pathlib import Path


class CSVLoader:
    """Load rows from a CSV file. Each row becomes one chunk."""

    def __init__(self, text_column: str = "text", min_words: int = 5):
        self.text_column = text_column
        self.min_words = min_words

    def load(self, source: str) -> list[dict]:
        docs = []
        with open(source, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get(self.text_column, "").strip()
                if len(text.split()) < self.min_words:
                    continue
                # All columns except the text column go into metadata
                meta = {k: v for k, v in row.items() if k != self.text_column}
                meta.update({"source": Path(source).name, "chunk_id": i})
                docs.append({"text": text, "metadata": meta})
        return docs
```

#### Step 2 — Register in the loader factory

Open `ingestion/registry.py` and add before the final `raise ValueError`:

```python
if loader_type == "csv":
    from ingestion.csv_loader import CSVLoader
    return CSVLoader(
        text_column=cfg.get("csv_text_column", "text"),
        min_words=cfg.get("csv_min_words", 5),
    )
```

#### Step 3 — Add a Literal type to the config validator

Open `config.py` and update the `loader` field:

```python
loader: Literal["pdf", "markdown", "jsonl", "html", "directory", "csv"] = "pdf"
```

Also add any new config keys:

```python
csv_text_column: str = "text"
csv_min_words: int = 5
```

#### Step 4 — Use in your datasource config

```json
{
  "loader":           "csv",
  "csv_text_column":  "body",
  "csv_min_words":    10
}
```

#### Step 5 — Write a test

```python
# tests/test_csv_loader.py
def test_csv_loader_reads_rows(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("body,category\nHello world from a CSV file with words,A\nShort,B\n")

    from ingestion.csv_loader import CSVLoader
    loader = CSVLoader(text_column="body", min_words=3)
    docs = loader.load(str(csv_file))

    assert len(docs) == 1             # "Short" row skipped (< 3 words)
    assert "Hello world" in docs[0]["text"]
    assert docs[0]["metadata"]["category"] == "A"
    assert docs[0]["metadata"]["chunk_id"] == 0
```

---

← [Back to FAQ index](../../FAQ.md)
