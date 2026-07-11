#!/usr/bin/env python3
"""
Generate two-column research paper PDF — KVForge.
Output: docs/KVForge_Research_Paper.pdf
"""

import re
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Image, Table, TableStyle,
    FrameBreak, NextPageTemplate, HRFlowable,
    KeepTogether, PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
PAPER_MD = ROOT / "KVForge_Research_Paper.md"
OUT_PDF  = ROOT / "KVForge_Research_Paper.pdf"
FONT_DIR = Path(
    "/usr/local/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf"
)

# ── Page geometry ──────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = letter          # 612 × 792 pt  (8.5 × 11 in)
MARGIN  = 0.72 * inch            # 51.84 pt
COL_GAP = 0.24 * inch            # 17.28 pt
BODY_W  = PAGE_W - 2 * MARGIN   # 508.32 pt
COL_W   = (BODY_W - COL_GAP) / 2  # 245.52 pt ≈ 3.41 in
BODY_H  = PAGE_H - 2 * MARGIN   # 688.32 pt ≈ 9.56 in

# First-page split: compact title strip (title + author + github) only
# Abstract and body go straight into two columns below
TITLE_H = 1.05 * inch           # full-width title block at top of page 1
TWO_H   = BODY_H - TITLE_H      # two-column section (abstract + body) on page 1

# ── Fonts ──────────────────────────────────────────────────────────────────────
F  = "DVS"    # DejaVu Serif
FM = "DVSM"   # DejaVu Sans Mono

pdfmetrics.registerFont(TTFont(F,        str(FONT_DIR / "DejaVuSerif.ttf")))
pdfmetrics.registerFont(TTFont(F+"B",    str(FONT_DIR / "DejaVuSerif-Bold.ttf")))
pdfmetrics.registerFont(TTFont(F+"I",    str(FONT_DIR / "DejaVuSerif-Italic.ttf")))
pdfmetrics.registerFont(TTFont(F+"BI",   str(FONT_DIR / "DejaVuSerif-BoldItalic.ttf")))
pdfmetrics.registerFont(TTFont(FM,       str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont(FM+"B",   str(FONT_DIR / "DejaVuSansMono-Bold.ttf")))
registerFontFamily(F, normal=F, bold=F+"B", italic=F+"I", boldItalic=F+"BI")

# ── Colours ────────────────────────────────────────────────────────────────────
C = dict(
    black   = colors.HexColor("#1F2937"),
    mid     = colors.HexColor("#374151"),
    gray    = colors.HexColor("#6B7280"),
    lgray   = colors.HexColor("#D1D5DB"),
    vlight  = colors.HexColor("#F9FAFB"),
    rule    = colors.HexColor("#374151"),
    code_bg = colors.HexColor("#F3F4F6"),
    stripe  = colors.HexColor("#F0F4FF"),
)

# ── Styles ─────────────────────────────────────────────────────────────────────
def _s(name, **kw):
    base = dict(fontName=F, fontSize=10, leading=13.5,
                textColor=C["black"], alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)

ST = dict(
    title   = _s("title",  fontName=F+"B", fontSize=16.5, leading=20.5,
                 alignment=TA_CENTER, spaceAfter=4),
    author  = _s("author", fontName=F+"B", fontSize=11.5, leading=14.5,
                 alignment=TA_CENTER, spaceAfter=0),
    affil   = _s("affil",  fontName=F+"I", fontSize=9.5, leading=12.5,
                 alignment=TA_CENTER, spaceAfter=0, textColor=C["mid"]),
    abtitle = _s("abtitle",fontName=F+"B", fontSize=10, leading=13,
                 alignment=TA_CENTER, spaceAfter=2),
    abstract= _s("abstract", fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
                 leftIndent=10, rightIndent=10),
    # Section headings — keepWithNext ensures header stays in same column as content
    h2      = _s("h2", fontName=F+"B", fontSize=11.5, leading=14,
                 spaceBefore=9, spaceAfter=3, alignment=TA_LEFT, keepWithNext=1),
    h3      = _s("h3", fontName=F+"B", fontSize=10.5, leading=13,
                 spaceBefore=6, spaceAfter=2, alignment=TA_LEFT, keepWithNext=1),
    h4      = _s("h4", fontName=F+"BI", fontSize=10, leading=13,
                 spaceBefore=4, spaceAfter=2, alignment=TA_LEFT, keepWithNext=1),
    body    = _s("body", spaceBefore=0, spaceAfter=3.5),
    bullet  = _s("bullet", leftIndent=11, firstLineIndent=0,
                 spaceBefore=1, spaceAfter=1),
    numlist = _s("numlist", leftIndent=14, firstLineIndent=-10,
                 spaceBefore=1.5, spaceAfter=1.5),
    # Code blocks — XML-safe, monospace, shaded
    code    = _s("code", fontName=FM, fontSize=7.4, leading=9.8,
                 alignment=TA_LEFT, textColor=C["black"],
                 backColor=C["code_bg"],
                 leftIndent=5, rightIndent=5,
                 spaceBefore=4, spaceAfter=4),
    caption = _s("caption", fontName=F+"I", fontSize=8.5, leading=11,
                 alignment=TA_CENTER, textColor=C["mid"],
                 spaceBefore=2, spaceAfter=5),
    quote   = _s("quote", fontName=F+"I", fontSize=9.5, leading=12.5,
                 leftIndent=10, rightIndent=10, textColor=C["mid"],
                 spaceBefore=3, spaceAfter=3),
    ref     = _s("ref", fontSize=8.5, leading=11.5, alignment=TA_LEFT,
                 leftIndent=14, firstLineIndent=-14, spaceBefore=1.5),
    tbl_hdr = _s("tbl_hdr", fontName=F+"B", fontSize=8, leading=10, alignment=TA_CENTER),
    tbl_cel = _s("tbl_cel", fontSize=8, leading=10, alignment=TA_CENTER),
    tbl_lft = _s("tbl_lft", fontSize=8, leading=10, alignment=TA_LEFT),
    tbl_sm  = _s("tbl_sm",  fontSize=6.8, leading=9, alignment=TA_CENTER),
    tbl_sml = _s("tbl_sml", fontSize=6.8, leading=9, alignment=TA_LEFT),
    tbl_smh = _s("tbl_smh", fontName=F+"B", fontSize=6.8, leading=9, alignment=TA_CENTER),
    footer  = _s("footer", fontSize=8, leading=10, alignment=TA_CENTER,
                 textColor=C["gray"]),
)

# ── XML-safe inline converter ──────────────────────────────────────────────────
_BOLD_RE   = re.compile(r'\*\*\*(.+?)\*\*\*')
_BOLD2_RE  = re.compile(r'\*\*(.+?)\*\*')
_ITA_RE    = re.compile(r'(?<!\*)\*([^*\n]+?)\*(?!\*)')
_CODE_RE   = re.compile(r'`([^`\n]+?)`')
_LINK_RE   = re.compile(r'\[([^\]]+)\]\([^)]+\)')

def _xml_escape(text: str) -> str:
    """Escape only &, <, > that are NOT already part of markup."""
    text = text.replace("&", "&amp;")
    # Escape < and > that would be mistaken for tags
    text = re.sub(r'<(?![/a-zA-Z])', '&lt;', text)
    text = re.sub(r'(?<![="\'-a-zA-Z])>', '&gt;', text)
    return text

def inline(text: str) -> str:
    """Convert inline markdown to ReportLab Paragraph XML."""
    t = _xml_escape(text)
    t = _BOLD_RE.sub(r'<b><i>\1</i></b>', t)
    t = _BOLD2_RE.sub(r'<b>\1</b>', t)
    t = _ITA_RE.sub(r'<i>\1</i>', t)
    t = _CODE_RE.sub(lambda m: f'<font name="DVSM" size="7.5" color="#374151">'
                               f'{m.group(1)}</font>', t)
    t = _LINK_RE.sub(r'\1', t)
    return t

def safe_code(text: str) -> str:
    """Escape text for use inside a code Paragraph (no markup, just safe XML)."""
    t = text.replace("&", "&amp;")
    t = t.replace("<", "&lt;")
    t = t.replace(">", "&gt;")
    return t

# ── Markdown block parser ──────────────────────────────────────────────────────
def parse_md(path: Path) -> list:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i, n = 0, len(lines)

    while i < n:
        raw = lines[i]
        s   = raw.strip()

        if not s:
            i += 1; continue

        # Headings
        if raw.startswith("#### "): blocks.append(("h4", raw[5:].strip())); i += 1; continue
        if raw.startswith("### "):  blocks.append(("h3", raw[4:].strip())); i += 1; continue
        if raw.startswith("## "):   blocks.append(("h2", raw[3:].strip())); i += 1; continue
        if raw.startswith("# ") and not raw.startswith("## "):
            blocks.append(("h1", raw[2:].strip())); i += 1; continue

        # Horizontal rule
        if s in ("---", "***", "___"):
            blocks.append(("hr", None)); i += 1; continue

        # Fenced code block
        if raw.startswith("```"):
            lang = raw[3:].strip()
            i += 1
            code = []
            while i < n and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            blocks.append(("code", (lang, "\n".join(code)))); continue

        # Image embed
        if raw.startswith("!["):
            m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', raw)
            if m:
                blocks.append(("image", (m.group(1), m.group(2))))
            i += 1; continue

        # Table
        if raw.startswith("|"):
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            blocks.append(("table", rows)); continue

        # Blockquote
        if raw.startswith("> "):
            bq = []
            while i < n and lines[i].startswith("> "):
                bq.append(lines[i][2:]); i += 1
            blocks.append(("blockquote", " ".join(bq))); continue

        # Bullet list
        if re.match(r'^[-*] ', raw):
            items = []
            while i < n and re.match(r'^[-*] ', lines[i]):
                items.append(lines[i][2:].strip()); i += 1
            blocks.append(("bullets", items)); continue

        # Numbered list
        if re.match(r'^\d+\. ', raw):
            items = []
            while i < n and re.match(r'^\d+\. ', lines[i]):
                items.append(re.sub(r'^\d+\. ', '', lines[i]).strip()); i += 1
            blocks.append(("numlist", items)); continue

        # Regular paragraph
        para = []
        while i < n:
            l = lines[i]
            if (not l.strip()
                    or l.startswith("#")
                    or l.strip() in ("---", "***", "___")
                    or l.startswith("```")
                    or l.startswith("![")
                    or l.startswith("|")
                    or l.startswith("> ")
                    or re.match(r'^[-*] ', l)
                    or re.match(r'^\d+\. ', l)):
                break
            para.append(l.rstrip()); i += 1
        if para:
            blocks.append(("para", " ".join(para)))

    return blocks

# ── Image helper ───────────────────────────────────────────────────────────────
def _image_fl(alt: str, rel_path: str, max_w: float) -> list:
    img_path = ROOT / rel_path
    if not img_path.exists():
        return [Paragraph(f"[Missing figure: {rel_path}]", ST["caption"])]
    try:
        with PILImage.open(img_path) as im:
            pw, ph = im.size
        scale = min(max_w / pw, (max_w * 1.8) / ph)
        iw, ih = pw * scale, ph * scale
        out = [Spacer(1, 3), Image(str(img_path), width=iw, height=ih)]
        if alt.strip():
            out.append(Paragraph(alt.strip(), ST["caption"]))
        out.append(Spacer(1, 3))
        return out
    except Exception as e:
        return [Paragraph(f"[Image error: {e}]", ST["caption"])]

# ── Table helper ───────────────────────────────────────────────────────────────
def _table_fl(rows: list, max_w: float) -> list:
    def parse_row(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    parsed = [parse_row(r) for r in rows]
    parsed = [r for r in parsed if not all(re.match(r'^[-:= ]+$', c) for c in r)]
    if not parsed:
        return []

    ncols = max(len(r) for r in parsed)
    for r in parsed:
        while len(r) < ncols: r.append("")

    # Choose small font for wide tables so they fit in column
    wide = ncols > 4
    hdr_st  = ST["tbl_smh"] if wide else ST["tbl_hdr"]
    cel_st  = ST["tbl_sm"]  if wide else ST["tbl_cel"]
    lft_st  = ST["tbl_sml"] if wide else ST["tbl_lft"]

    # Column widths
    if ncols == 1:
        cws = [max_w]
    elif ncols == 2:
        cws = [max_w * 0.44, max_w * 0.56]
    elif ncols == 3:
        cws = [max_w * 0.32, max_w * 0.36, max_w * 0.32]
    elif ncols == 4:
        cws = [max_w * 0.30, max_w * 0.24, max_w * 0.24, max_w * 0.22]
    elif ncols == 5:
        first = max_w * 0.24
        rest  = (max_w - first) / 4
        cws   = [first] + [rest] * 4
    else:
        # 6+ cols: very narrow, rely on word-wrap
        first = max_w * 0.22
        rest  = (max_w - first) / (ncols - 1)
        cws   = [first] + [rest] * (ncols - 1)

    tdata = []
    for ri, row in enumerate(parsed):
        style = hdr_st if ri == 0 else cel_st
        if ri > 0:
            tdata.append(
                [Paragraph(inline(row[0]), lft_st)]
                + [Paragraph(inline(c), cel_st if not wide else ST["tbl_sm"]) for c in row[1:]]
            )
        else:
            tdata.append([Paragraph(inline(c), hdr_st) for c in row])

    nrows = len(tdata)
    ts = TableStyle([
        ("FONTNAME",       (0, 0), (-1,  0),  F+"B"),
        ("LINEABOVE",      (0, 0), (-1,  0),  1.2, C["black"]),
        ("LINEBELOW",      (0, 0), (-1,  0),  0.7, C["black"]),
        ("LINEBELOW",      (0, nrows-1), (-1, nrows-1), 1.2, C["black"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),  [colors.white, C["stripe"]]),
        ("TOPPADDING",     (0, 0), (-1, -1),  2.5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1),  2.5),
        ("LEFTPADDING",    (0, 0), (-1, -1),  3),
        ("RIGHTPADDING",   (0, 0), (-1, -1),  3),
        ("VALIGN",         (0, 0), (-1, -1),  "TOP"),
    ])
    tbl = Table(tdata, colWidths=cws, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(ts)
    return [Spacer(1, 4), tbl, Spacer(1, 6)]

# ── Block → flowable converter ─────────────────────────────────────────────────
def blocks_to_flowables(blocks: list, col_w: float, body_st=None) -> list:
    if body_st is None:
        body_st = ST["body"]
    out = []

    for kind, payload in blocks:
        if kind == "h1":
            out += [Paragraph(payload, ST["title"]), Spacer(1, 4)]

        elif kind in ("h2", "h3", "h4"):
            # keepWithNext=1 in the style ensures header stays in same
            # column/page as following content
            out += [Spacer(1, 3), Paragraph(payload, ST[kind])]

        elif kind == "hr":
            out += [Spacer(1, 3),
                    HRFlowable(width="100%", thickness=0.6, color=C["rule"]),
                    Spacer(1, 3)]

        elif kind == "para":
            out.append(Paragraph(inline(payload), body_st))

        elif kind == "blockquote":
            out.append(Paragraph(inline(payload), ST["quote"]))

        elif kind == "bullets":
            for item in payload:
                out.append(Paragraph(f"• {inline(item)}", ST["bullet"]))

        elif kind == "numlist":
            for idx, item in enumerate(payload, 1):
                out.append(Paragraph(
                    f"<b>{idx}.</b>&nbsp;&nbsp;{inline(item)}", ST["numlist"]))

        elif kind == "code":
            _, text = payload
            escaped = safe_code(text).replace("\n", "<br/>")
            out.append(
                Paragraph(f'<font name="DVSM" size="7.4">{escaped}</font>',
                           ST["code"]))

        elif kind == "image":
            alt, rel = payload
            out += _image_fl(alt, rel, col_w)

        elif kind == "table":
            out += _table_fl(payload, col_w)

    return out

# ── Header section handler ─────────────────────────────────────────────────────
def header_flowables(hdr_blocks: list) -> list:
    """Produce full-width title/author/abstract flowables."""
    out = []
    for kind, payload in hdr_blocks:
        if kind == "h1":
            out += [Spacer(1, 4), Paragraph(payload, ST["title"]), Spacer(1, 5)]

        elif kind == "para":
            text = payload.strip()
            # Author block: bold name + affiliation lines joined
            if text.startswith("**"):
                # Split on two trailing spaces (markdown hard-break)
                parts = re.split(r'  +', text)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    # Bold name
                    if part.startswith("**"):
                        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', part)
                        out.append(Paragraph(clean, ST["author"]))
                    else:
                        out.append(Paragraph(inline(part), ST["affil"]))
                out.append(Spacer(1, 2))
            elif "Independent Research" in text:
                out.append(Paragraph(text, ST["affil"]))
            else:
                out.append(Paragraph(inline(text), ST["abstract"]))

        elif kind == "hr":
            out += [Spacer(1, 3),
                    HRFlowable(width="100%", thickness=0.7, color=C["rule"]),
                    Spacer(1, 3)]

        elif kind == "h2" and payload.strip().lower() == "abstract":
            out += [Spacer(1, 4), Paragraph("Abstract", ST["abtitle"]), Spacer(1, 2)]

        elif kind == "h2":
            pass  # skip other h2s in header section

    return out

# ── Page number callback ───────────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 8)
    canvas.setFillColor(C["gray"])
    txt = f"KVForge — Dr. Hemant Joshi · 2025"
    canvas.drawString(MARGIN, MARGIN / 2 - 2, txt)
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN / 2 - 2, f"Page {doc.page}")
    # Vertical column separator line on pages 2+ (page 1 has the HDR frame)
    if doc.page > 1:
        sep_x = MARGIN + COL_W + COL_GAP / 2
        canvas.setStrokeColor(C["lgray"])
        canvas.setLineWidth(0.5)
        canvas.line(sep_x, MARGIN, sep_x, PAGE_H - MARGIN)
    canvas.restoreState()

def _footer_p1(canvas, doc):
    _footer(canvas, doc)
    # Draw a thin separator between header block and two-column section
    sep_y = MARGIN + TWO_H
    canvas.saveState()
    canvas.setStrokeColor(C["lgray"])
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, sep_y, PAGE_W - MARGIN, sep_y)
    # Vertical column sep in the two-column portion of page 1
    sep_x = MARGIN + COL_W + COL_GAP / 2
    canvas.setStrokeColor(C["lgray"])
    canvas.setLineWidth(0.5)
    canvas.line(sep_x, MARGIN, sep_x, sep_y)
    canvas.restoreState()

# ── Build PDF ──────────────────────────────────────────────────────────────────
def build_pdf(blocks: list):
    doc = BaseDocTemplate(
        str(OUT_PDF), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    # ── Page 1: full-width header + two-column body below ─────────────────────
    hdr_frame = Frame(
        MARGIN, MARGIN + TWO_H, BODY_W, HDR_H,
        id="hdr",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    p1_left = Frame(
        MARGIN, MARGIN, COL_W, TWO_H,
        id="p1l",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    p1_right = Frame(
        MARGIN + COL_W + COL_GAP, MARGIN, COL_W, TWO_H,
        id="p1r",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    page1_tpl = PageTemplate(
        id="page1",
        frames=[hdr_frame, p1_left, p1_right],
        onPage=_footer_p1,
    )

    # ── Subsequent pages: two-column only ─────────────────────────────────────
    left_frame = Frame(
        MARGIN, MARGIN, COL_W, BODY_H,
        id="left",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    right_frame = Frame(
        MARGIN + COL_W + COL_GAP, MARGIN, COL_W, BODY_H,
        id="right",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    main_tpl = PageTemplate(
        id="main",
        frames=[left_frame, right_frame],
        onPage=_footer,
    )

    doc.addPageTemplates([page1_tpl, main_tpl])

    # ── Split blocks at first numbered section ────────────────────────────────
    hdr_blocks, body_blocks = [], []
    in_body = False
    for b in blocks:
        kind, payload = b
        if not in_body and kind == "h2" and re.match(r'^1[.\s]', payload):
            in_body = True
        (body_blocks if in_body else hdr_blocks).append(b)

    # ── Assemble story ────────────────────────────────────────────────────────
    hdr_fl   = header_flowables(hdr_blocks)
    body_fl  = blocks_to_flowables(body_blocks, COL_W)

    # Signal: after header content overflows into hdr_frame, break to p1_left,
    # then subsequent pages use "main" template
    story = (
        hdr_fl
        + [FrameBreak(), NextPageTemplate("main")]
        + body_fl
    )

    doc.build(story)
    size_kb = OUT_PDF.stat().st_size // 1024
    print(f"✓  {OUT_PDF.name}  —  {size_kb} KB")

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Parsing markdown …")
    blocks = parse_md(PAPER_MD)
    print(f"  {len(blocks)} blocks")
    print("Rendering PDF …")
    build_pdf(blocks)
