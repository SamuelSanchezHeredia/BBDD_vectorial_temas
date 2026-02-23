"""Script autónomo para probar el chunking semántico con el PDF real."""
import re
import fitz
import os

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saberes_2ESO.pdf")
MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 60
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks_preview.txt")


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    doc.close()
    return pages


def is_heading(line):
    line = line.strip()
    if not line or len(line) > 60:
        return False
    patterns = [
        r"^(Lengua Castellana|Matemáticas|Ciencias Naturales|Ciencias Sociales|"
        r"Historia|Geografía e Historia|Biología|Física y Química|"
        r"Inglés|Francés|Segunda Lengua|Educación Física|Tecnología|"
        r"Música|Plástica|Artes Plásticas|Religión|Ética|Filosofía|"
        r"Economía|Informática|Latín|Literatura|Valores Cívicos|"
        r"Educación en Valores)",
        r"^Saberes básicos",
    ]
    for p in patterns:
        if re.match(p, line, re.IGNORECASE):
            return True
    return False


def split_by_sentences(text, max_chars):
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    fragments, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                fragments.append(current)
            current = s
    if current:
        fragments.append(current)
    return fragments


def split_into_chunks(pages):
    chunks = []

    def flush(text, section, page):
        text = text.strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append({"text": text, "section": section, "page": page})

    current_section = "General"
    current_chunk = ""
    current_page = 1

    for page_data in pages:
        page_num = page_data["page"]
        lines = page_data["text"].split("\n")
        para_buf = ""
        for line in lines:
            s = line.strip()
            if not s:
                if para_buf.strip():
                    para = para_buf.strip()
                    para_buf = ""
                    if len(current_chunk) + len(para) + 2 <= MAX_CHUNK_CHARS:
                        current_chunk = (current_chunk + "\n\n" + para).strip()
                        current_page = page_num
                    else:
                        flush(current_chunk, current_section, current_page)
                        if len(para) > MAX_CHUNK_CHARS:
                            for f in split_by_sentences(para, MAX_CHUNK_CHARS):
                                flush(f, current_section, page_num)
                            current_chunk = ""
                        else:
                            current_chunk = para
                        current_page = page_num
                continue
            if is_heading(s):
                if para_buf.strip():
                    para = para_buf.strip()
                    para_buf = ""
                    current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
                flush(current_chunk, current_section, current_page)
                current_chunk = ""
                current_section = s
                current_page = page_num
            else:
                para_buf = (para_buf + " " + s).strip()
        if para_buf.strip():
            para = para_buf.strip()
            if len(current_chunk) + len(para) + 2 <= MAX_CHUNK_CHARS:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                flush(current_chunk, current_section, current_page)
                if len(para) > MAX_CHUNK_CHARS:
                    for f in split_by_sentences(para, MAX_CHUNK_CHARS):
                        flush(f, current_section, page_num)
                    current_chunk = ""
                else:
                    current_chunk = para
            current_page = page_num
    flush(current_chunk, current_section, current_page)
    return chunks


if __name__ == "__main__":
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = split_into_chunks(pages)
    sections = sorted(set(c["section"] for c in chunks))

    lines = []
    lines.append(f"Total chunks: {len(chunks)}")
    lines.append(f"Secciones ({len(sections)}): {', '.join(sections)}")
    lines.append("")
    for i, c in enumerate(chunks):
        lines.append(
            f"--- Chunk {i+1} | Seccion: {c['section']} | Pag: {c['page']} | {len(c['text'])} chars ---"
        )
        lines.append(c["text"])
        lines.append("")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



