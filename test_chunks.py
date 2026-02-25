"""Script autónomo para probar el chunking semántico con el PDF real."""
import re
import fitz
import os

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saberes_2ESO.pdf")
MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 20
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


def detect_trimester(text):
    """Detecta si el texto contiene una marca de trimestre."""
    m = re.match(r"^\s*(\d)\.?[°ºo]?\s*trimestre\b", text, re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"{n}.º trimestre"
    return None


def split_by_trimesters(text):
    """Divide texto con marcas de trimestre inline en [(trimestre, contenido), ...]."""
    pattern = r"(\d\.?[°ºo]?\s*trimestre)\s*"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        return [("General", text.strip())]
    result = []
    pre_text = parts[0].strip()
    if pre_text:
        result.append(("General", pre_text))
    for i in range(1, len(parts), 2):
        raw_marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        m = re.match(r"(\d)", raw_marker)
        tri_name = f"{m.group(1)}.º trimestre" if m else raw_marker
        result.append((tri_name, content))
    return result


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

    def flush(text, section, trimester, page):
        text = text.strip()
        if len(text) >= MIN_CHUNK_CHARS:
            prefix = f"[{section}]" if section != "General" else ""
            if trimester != "General":
                prefix += f" [{trimester}]" if prefix else f"[{trimester}]"
            enriched = f"{prefix} {text}".strip() if prefix else text
            chunks.append({"text": enriched, "section": section, "trimester": trimester, "page": page})

    def process_paragraph(para, section, trimester, page, current_chunk):
        """Procesa un párrafo dividiéndolo por trimestres si los contiene."""
        tri_parts = split_by_trimesters(para)
        if len(tri_parts) == 1 and tri_parts[0][0] == "General":
            text = tri_parts[0][1]
            if len(current_chunk) + len(text) + 2 <= MAX_CHUNK_CHARS:
                current_chunk = (current_chunk + "\n\n" + text).strip()
            else:
                flush(current_chunk, section, trimester, page)
                if len(text) > MAX_CHUNK_CHARS:
                    for f in split_by_sentences(text, MAX_CHUNK_CHARS):
                        flush(f, section, trimester, page)
                    current_chunk = ""
                else:
                    current_chunk = text
            return current_chunk, trimester
        # Hay marcas de trimestre inline
        flush(current_chunk, section, trimester, page)
        current_chunk = ""
        last_tri = trimester
        for tri_name, tri_content in tri_parts:
            if tri_name != "General":
                last_tri = tri_name
            if tri_content:
                if len(tri_content) > MAX_CHUNK_CHARS:
                    for f in split_by_sentences(tri_content, MAX_CHUNK_CHARS):
                        flush(f, section, last_tri, page)
                else:
                    flush(tri_content, section, last_tri, page)
        return "", last_tri

    current_section = "General"
    current_trimester = "General"
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
                    current_chunk, current_trimester = process_paragraph(
                        para, current_section, current_trimester, page_num, current_chunk
                    )
                    current_page = page_num
                continue
            if is_heading(s):
                if para_buf.strip():
                    para = para_buf.strip()
                    para_buf = ""
                    current_chunk, current_trimester = process_paragraph(
                        para, current_section, current_trimester, page_num, current_chunk
                    )
                flush(current_chunk, current_section, current_trimester, current_page)
                current_chunk = ""
                current_section = s
                current_trimester = "General"
                current_page = page_num
            else:
                para_buf = (para_buf + " " + s).strip()
        if para_buf.strip():
            para = para_buf.strip()
            current_chunk, current_trimester = process_paragraph(
                para, current_section, current_trimester, page_num, current_chunk
            )
            current_page = page_num
    flush(current_chunk, current_section, current_trimester, current_page)
    return chunks


if __name__ == "__main__":
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = split_into_chunks(pages)
    sections = sorted(set(c["section"] for c in chunks))
    trimesters = sorted(set(c["trimester"] for c in chunks))

    lines = []
    lines.append(f"Total chunks: {len(chunks)}")
    lines.append(f"Secciones ({len(sections)}): {', '.join(sections)}")
    lines.append(f"Trimestres ({len(trimesters)}): {', '.join(trimesters)}")
    lines.append("")
    for i, c in enumerate(chunks):
        lines.append(
            f"--- Chunk {i+1} | Seccion: {c['section']} | Trimestre: {c['trimester']} | Pag: {c['page']} | {len(c['text'])} chars ---"
        )
        lines.append(c["text"])
        lines.append("")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



