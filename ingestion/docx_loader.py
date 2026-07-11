import hashlib
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn


def _heading_level(paragraph) -> int | None:
    style_name = paragraph.style.name
    if style_name.startswith("Heading"):
        try:
            return int(style_name.split()[-1])
        except ValueError:
            return None
    return None


def _section_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocxLoader:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, source: str) -> list[dict]:
        doc = Document(source)
        props = doc.core_properties
        path = Path(source)

        current_heading_text = ""
        current_heading_level = None
        section_paragraphs: list[str] = []
        chunks: list[dict] = []
        chunk_id = 0

        def flush_section():
            nonlocal chunk_id
            if not section_paragraphs:
                return
            full_text = "\n".join(section_paragraphs)
            words = full_text.split()
            step = max(self.chunk_size - self.chunk_overlap, 1)
            start = 0
            while start < len(words):
                window = words[start: start + self.chunk_size]
                chunk_text = " ".join(window)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "heading_text": current_heading_text,
                        "heading_level": current_heading_level,
                        "is_table": False,
                        "author": props.author or "",
                        "modified": props.modified.isoformat() if props.modified else "",
                        "source": path.name,
                        "chunk_id": chunk_id,
                        "section_hash": _section_hash(full_text),
                    },
                })
                chunk_id += 1
                start += step

        body = doc.element.body
        for child in body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "p":
                from docx.text.paragraph import Paragraph
                para = Paragraph(child, doc)
                text = para.text.strip()
                if not text:
                    continue
                level = _heading_level(para)
                if level is not None:
                    flush_section()
                    section_paragraphs.clear()
                    current_heading_text = text
                    current_heading_level = level
                else:
                    section_paragraphs.append(text)
            elif tag == "tbl":
                flush_section()
                section_paragraphs.clear()
                from docx.table import Table
                table = Table(child, doc)
                if not table.rows:
                    continue
                headers = [cell.text.strip() for cell in table.rows[0].cells]
                # Compute table_hash over all rows' formatted text joined with newlines
                all_row_texts = []
                for row in table.rows[1:]:
                    cells = [cell.text.strip() for cell in row.cells]
                    row_text = " | ".join(
                        f"{h}: {v}" for h, v in zip(headers, cells)
                    )
                    all_row_texts.append(row_text)
                table_hash = _section_hash("\n".join(all_row_texts))

                for row_idx, row in enumerate(table.rows[1:], start=1):
                    cells = [cell.text.strip() for cell in row.cells]
                    row_text = " | ".join(
                        f"{h}: {v}" for h, v in zip(headers, cells)
                    )
                    chunks.append({
                        "text": row_text,
                        "metadata": {
                            "heading_text": current_heading_text,
                            "heading_level": current_heading_level,
                            "is_table": True,
                            "table_position": {"row": row_idx, "col": 0},
                            "author": props.author or "",
                            "modified": props.modified.isoformat() if props.modified else "",
                            "source": path.name,
                            "chunk_id": chunk_id,
                            "section_hash": table_hash,
                        },
                    })
                    chunk_id += 1

        flush_section()
        return chunks
