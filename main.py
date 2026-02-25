"""
Base de datos vectorial con Pinecone + FAISS + HuggingFace (sentence-transformers).
Lee el PDF Saberes_2ESO.pdf, genera embeddings y los almacena en Pinecone (núcleo)
y FAISS (motor de búsqueda local).

Arquitectura híbrida:
  - Pinecone: almacenamiento persistente y centralizado (fuente de verdad).
  - FAISS: búsqueda local ultrarrápida y offline.

Estrategia de chunking semántico:
  1. Detecta secciones por encabezados (asignaturas, trimestres, títulos).
  2. Dentro de cada sección agrupa párrafos hasta alcanzar MAX_CHUNK_CHARS.
  3. Si un párrafo supera MAX_CHUNK_CHARS, lo divide por oraciones completas.
  4. Nunca corta palabras a mitad. Cada chunk hereda sección y página de origen.
"""

import os
import re
import sys
import time
import json
import fitz  # PyMuPDF
import numpy as np
import faiss
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
INDEX_NAME = "saberes-2eso"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims, multilingüe (español)
EMBEDDING_DIM = 384
MAX_CHUNK_CHARS = 800   # máximo de caracteres por chunk
MIN_CHUNK_CHARS = 20    # descartar chunks demasiado pequeños (ruido)
PDF_PATH = os.path.join(os.path.dirname(__file__), "Saberes_2ESO.pdf")
FAISS_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
FAISS_METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")


# ──────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────
def load_env():
    """Carga y valida las variables de entorno."""
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key.startswith("tu_clave"):
        print("❌ ERROR: Falta la PINECONE_API_KEY en el archivo .env")
        sys.exit(1)
    return api_key


def extract_text_from_pdf(path: str) -> list[dict]:
    """Extrae texto del PDF página a página usando PyMuPDF."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    doc.close()
    return pages


def is_heading(line: str) -> bool:
    """
    Detecta si una línea es un encabezado de sección (asignatura o título principal).
    Solo marca como heading los nombres de asignaturas y títulos del documento,
    NO los sub-apartados como '1.º trimestre' que son contenido dentro de la sección.
    """
    line = line.strip()
    if not line or len(line) > 60:
        return False
    # Nombres de asignaturas conocidas
    patterns = [
        r"^(Lengua Castellana|Matemáticas|Ciencias Naturales|Ciencias Sociales|"
        r"Historia|Geografía e Historia|Biología|Física y Química|"
        r"Inglés|Francés|Segunda Lengua|Educación Física|Tecnología|"
        r"Música|Plástica|Artes Plásticas|Religión|Ética|Filosofía|"
        r"Economía|Informática|Latín|Literatura|Valores Cívicos|"
        r"Educación en Valores)",
        r"^Saberes básicos",                          # título principal del documento
    ]
    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def detect_trimester(text: str) -> str | None:
    """
    Detecta si el texto EMPIEZA con una marca de trimestre (p.ej. '1.º trimestre').
    Retorna '1.º trimestre', '2.º trimestre', '3.º trimestre' o None.
    """
    m = re.match(r"^\s*(\d)\.?[°ºo]?\s*trimestre\b", text, re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"{n}.º trimestre"
    return None


def split_by_trimesters(text: str) -> list[tuple[str, str]]:
    """
    Divide un texto que contiene marcas de trimestre inline en fragmentos.
    Retorna lista de tuplas (nombre_trimestre, contenido).
    Si no hay marcas de trimestre, retorna [("General", texto_completo)].
    Ejemplo:
      '1.º trimestre Números ... 2.º trimestre Álgebra ...'
      → [('1.º trimestre', 'Números ...'), ('2.º trimestre', 'Álgebra ...')]
    """
    # Patrón que captura las marcas de trimestre inline
    pattern = r"(\d\.?[°ºo]?\s*trimestre)\s*"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    # parts alterna: [texto_antes, marca1, texto1, marca2, texto2, ...]

    if len(parts) <= 1:
        # Sin marcas de trimestre
        return [("General", text.strip())]

    result = []
    # Si hay texto antes de la primera marca, se asigna como "General"
    pre_text = parts[0].strip()
    if pre_text:
        result.append(("General", pre_text))

    # Recorrer pares (marca, contenido)
    for i in range(1, len(parts), 2):
        raw_marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Normalizar nombre del trimestre
        m = re.match(r"(\d)", raw_marker)
        trimester_name = f"{m.group(1)}.º trimestre" if m else raw_marker
        result.append((trimester_name, content))

    return result


def split_by_sentences(text: str, max_chars: int) -> list[str]:
    """
    Divide un texto largo en fragmentos respetando oraciones completas.
    Corta por '. ', '.\n', '? ', '! ' sin partir palabras.
    """
    # Separar por fin de oración manteniendo el separador
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    fragments = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                fragments.append(current)
            # Si la oración sola supera el límite, la añade de todas formas
            # (no se puede dividir sin perder sentido)
            current = sentence
    if current:
        fragments.append(current)
    return fragments


def split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Chunking semántico en 4 niveles:
      1. Detecta encabezados → marca inicio de nueva sección (asignatura).
      2. Detecta marcas de trimestre (inline o al inicio) → chunk por trimestre.
      3. Agrupa párrafos del mismo trimestre/sección hasta MAX_CHUNK_CHARS.
      4. Si un párrafo supera MAX_CHUNK_CHARS, lo divide por oraciones.
    Cada chunk incluye: texto, página, sección, trimestre.
    """
    chunks = []

    def flush_chunk(text: str, section: str, trimester: str, page: int):
        """Añade el chunk a la lista si supera el mínimo de caracteres."""
        text = text.strip()
        if len(text) >= MIN_CHUNK_CHARS:
            # Prefijar contexto de sección/trimestre para mejor embedding
            prefix = f"[{section}]" if section != "General" else ""
            if trimester != "General":
                prefix += f" [{trimester}]" if prefix else f"[{trimester}]"
            enriched_text = f"{prefix} {text}".strip() if prefix else text
            chunks.append({
                "text": enriched_text,
                "section": section,
                "trimester": trimester,
                "page": page,
            })

    def process_paragraph(para: str, section: str, trimester: str, page: int,
                          current_chunk: str) -> tuple[str, str]:
        """
        Procesa un párrafo: si contiene marcas de trimestre inline, divide
        y emite chunks separados. Retorna (chunk_acumulado, trimestre_actual).
        """
        tri_parts = split_by_trimesters(para)

        # Si no hay trimestres, el párrafo es texto normal
        if len(tri_parts) == 1 and tri_parts[0][0] == "General":
            text = tri_parts[0][1]
            if len(current_chunk) + len(text) + 2 <= MAX_CHUNK_CHARS:
                current_chunk = (current_chunk + "\n\n" + text).strip()
            else:
                flush_chunk(current_chunk, section, trimester, page)
                if len(text) > MAX_CHUNK_CHARS:
                    for frag in split_by_sentences(text, MAX_CHUNK_CHARS):
                        flush_chunk(frag, section, trimester, page)
                    current_chunk = ""
                else:
                    current_chunk = text
            return current_chunk, trimester

        # Hay marcas de trimestre: flush lo acumulado y emitir un chunk por trimestre
        flush_chunk(current_chunk, section, trimester, page)
        current_chunk = ""
        last_tri = trimester

        for tri_name, tri_content in tri_parts:
            if tri_name != "General":
                last_tri = tri_name
            if tri_content:
                if len(tri_content) > MAX_CHUNK_CHARS:
                    for frag in split_by_sentences(tri_content, MAX_CHUNK_CHARS):
                        flush_chunk(frag, section, last_tri, page)
                else:
                    flush_chunk(tri_content, section, last_tri, page)

        return "", last_tri

    current_section = "General"
    current_trimester = "General"
    current_chunk = ""
    current_page = 1

    for page_data in pages:
        page_num = page_data["page"]
        lines = page_data["text"].split("\n")
        paragraph_buffer = ""

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Línea vacía → fin de párrafo
                if paragraph_buffer.strip():
                    para = paragraph_buffer.strip()
                    paragraph_buffer = ""
                    current_chunk, current_trimester = process_paragraph(
                        para, current_section, current_trimester, page_num, current_chunk
                    )
                    current_page = page_num
                continue

            if is_heading(stripped):
                # Guardar lo acumulado antes de cambiar de sección
                if paragraph_buffer.strip():
                    para = paragraph_buffer.strip()
                    paragraph_buffer = ""
                    current_chunk, current_trimester = process_paragraph(
                        para, current_section, current_trimester, page_num, current_chunk
                    )
                flush_chunk(current_chunk, current_section, current_trimester, current_page)
                current_chunk = ""
                current_section = stripped
                current_trimester = "General"
                current_page = page_num
            else:
                paragraph_buffer = (paragraph_buffer + " " + stripped).strip()

        # Al acabar la página, volcar el buffer restante
        if paragraph_buffer.strip():
            para = paragraph_buffer.strip()
            current_chunk, current_trimester = process_paragraph(
                para, current_section, current_trimester, page_num, current_chunk
            )
            current_page = page_num

    # Volcar el último chunk pendiente
    flush_chunk(current_chunk, current_section, current_trimester, current_page)

    return chunks


def save_faiss_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    """
    Construye un índice FAISS IndexFlatIP (producto interno ≈ coseno con
    vectores normalizados) y lo guarda en disco junto con los metadatos.
    """
    os.makedirs(FAISS_DIR, exist_ok=True)

    # Normalizar vectores para que el producto interno equivalga a similitud coseno
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Guardar metadatos (texto, página, sección, trimestre) indexados por posición
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": f"chunk-{i}",
            "text": chunk["text"],
            "page": chunk["page"],
            "section": chunk["section"],
            "trimester": chunk.get("trimester", "General"),
        })
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"   💾 Índice FAISS guardado en {FAISS_DIR}/ ({index.ntotal} vectores)")


def load_faiss_index():
    """
    Carga el índice FAISS y los metadatos desde disco.
    Retorna (index, metadata) o (None, None) si no existen los archivos.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        return None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def query_faiss(question: str, top_k: int = 5):
    """Busca en el índice FAISS local. Retorna resultados o None si no hay índice."""
    index, metadata = load_faiss_index()
    if index is None:
        return None

    model = SentenceTransformer(EMBEDDING_MODEL)
    q_embedding = model.encode(question)
    q_vector = np.array([q_embedding], dtype="float32")
    faiss.normalize_L2(q_vector)

    scores, indices = index.search(q_vector, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "score": float(score),
            "id": meta["id"],
            "metadata": {
                "text": meta["text"],
                "page": meta["page"],
                "section": meta["section"],
                "trimester": meta.get("trimester", "General"),
            },
        })
    return results


def sync():
    """
    Descarga todos los vectores de Pinecone y reconstruye el índice FAISS local.
    Útil cuando Pinecone fue actualizado desde otro entorno.
    """
    api_key = load_env()

    print("🌲 Conectando con Pinecone...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    # Obtener estadísticas del índice
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    if total_vectors == 0:
        print("⚠️  El índice de Pinecone está vacío. Ejecuta 'ingest' primero.")
        return

    print(f"   → {total_vectors} vectores en Pinecone.")

    # Descargar todos los vectores de Pinecone usando list + fetch
    print("📥 Descargando vectores de Pinecone...")
    all_ids = []
    for ids_batch in index.list():
        all_ids.extend(ids_batch)

    fetched = index.fetch(ids=all_ids)
    vectors_dict = fetched.vectors

    # Ordenar por ID para mantener consistencia
    sorted_ids = sorted(vectors_dict.keys(), key=lambda x: int(x.split("-")[1]))

    embeddings = []
    chunks = []
    for vid in sorted_ids:
        vec_data = vectors_dict[vid]
        embeddings.append(vec_data.values)
        chunks.append({
            "text": vec_data.metadata.get("text", ""),
            "page": vec_data.metadata.get("page", 0),
            "section": vec_data.metadata.get("section", ""),
            "trimester": vec_data.metadata.get("trimester", "General"),
        })

    embeddings_np = np.array(embeddings, dtype="float32")
    save_faiss_index(embeddings_np, chunks)
    print(f"\n✅ Sincronización completada: {len(chunks)} vectores descargados y guardados en FAISS.")


def create_or_get_index(pc: Pinecone) -> None:
    """Crea el índice en Pinecone si no existe y espera a que esté listo."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"   → Creando índice '{INDEX_NAME}' ({EMBEDDING_DIM} dimensiones)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("   ⏳ Esperando a que el índice esté listo...")
            time.sleep(2)
        print(f"   ✅ Índice creado.")
    else:
        print(f"   ℹ️  El índice '{INDEX_NAME}' ya existe.")


# ──────────────────────────────────────────────
# Ingesta
# ──────────────────────────────────────────────
def ingest():
    """Lee el PDF, genera embeddings y los sube a Pinecone."""
    api_key = load_env()

    # 1. Leer PDF
    print(f"📄 Leyendo PDF: {PDF_PATH}")
    pages = extract_text_from_pdf(PDF_PATH)
    print(f"   → {len(pages)} páginas con texto.")

    # 2. Chunking semántico
    print(f"✂️  Aplicando chunking semántico (max={MAX_CHUNK_CHARS} chars por chunk)...")
    chunks = split_into_chunks(pages)
    print(f"   → {len(chunks)} fragmentos generados.")

    # Mostrar resumen de secciones detectadas
    sections = sorted(set(c["section"] for c in chunks))
    print(f"   → {len(sections)} secciones detectadas: {', '.join(sections)}")

    # 3. Generar embeddings con HuggingFace
    print(f"🤗 Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"🔢 Generando embeddings para {len(texts)} fragmentos...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # 4. Conectar con Pinecone y crear índice
    print(f"🌲 Conectando con Pinecone...")
    pc = Pinecone(api_key=api_key)
    create_or_get_index(pc)
    index = pc.Index(INDEX_NAME)

    # 5. Borrar vectores anteriores para evitar duplicados en reingesta
    print(f"🗑️  Limpiando índice anterior...")
    index.delete(delete_all=True)

    # 6. Subir vectores en lotes
    print(f"🚀 Subiendo vectores a Pinecone...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(chunks))):
            vector_id = f"chunk-{j}"
            metadata = {
                "text":      chunks[j]["text"],
                "page":      chunks[j]["page"],
                "section":   chunks[j]["section"],
                "trimester": chunks[j].get("trimester", "General"),
            }
            batch.append((vector_id, embeddings[j].tolist(), metadata))
        index.upsert(vectors=batch)
        print(f"   → Subidos {min(i + batch_size, len(chunks))}/{len(chunks)}")

    # 7. Guardar índice FAISS local
    print(f"💾 Construyendo índice FAISS local...")
    save_faiss_index(embeddings, chunks)

    print(f"\n✅ Ingesta completada: {len(chunks)} fragmentos en el índice '{INDEX_NAME}' (Pinecone + FAISS local).")


# ──────────────────────────────────────────────
# Consulta
# ──────────────────────────────────────────────
def query(question: str, top_k: int = 5, engine: str = "auto"):
    """
    Busca los fragmentos más relevantes para una pregunta.

    Motores de búsqueda:
      - 'faiss':    Búsqueda local con FAISS (rápida, offline).
      - 'pinecone': Búsqueda en Pinecone (cloud).
      - 'auto':     Intenta FAISS primero, fallback a Pinecone.
    """
    used_engine = engine

    if engine in ("faiss", "auto"):
        faiss_results = query_faiss(question, top_k)
        if faiss_results is not None:
            used_engine = "faiss"
            print(f"\n🔎 Pregunta: {question}")
            print(f"⚡ Motor: FAISS (local)")
            print(f"📊 Top {top_k} resultados:\n")
            for i, match in enumerate(faiss_results, 1):
                score     = match["score"]
                text      = match["metadata"]["text"]
                page      = match["metadata"]["page"]
                section   = match["metadata"].get("section", "—")
                trimester = match["metadata"].get("trimester", "—")
                print(f"  [{i}] (similitud: {score:.4f}) — {section} | {trimester} | Página {page}")
                print(f"      {text[:300]}...")
                print()
            return faiss_results
        elif engine == "faiss":
            print("❌ No se encontró índice FAISS local. Ejecuta 'ingest' o 'sync' primero.")
            sys.exit(1)
        else:
            print("⚠️  No hay índice FAISS local, usando Pinecone como fallback...")

    # Búsqueda con Pinecone
    api_key = load_env()
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_embedding = model.encode(question).tolist()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    results = index.query(vector=q_embedding, top_k=top_k, include_metadata=True)

    print(f"\n🔎 Pregunta: {question}")
    print(f"🌲 Motor: Pinecone (cloud)")
    print(f"📊 Top {top_k} resultados:\n")
    for i, match in enumerate(results["matches"], 1):
        score     = match["score"]
        text      = match["metadata"]["text"]
        page      = match["metadata"]["page"]
        section   = match["metadata"].get("section", "—")
        trimester = match["metadata"].get("trimester", "—")
        print(f"  [{i}] (similitud: {score:.4f}) — {section} | {trimester} | Página {page}")
        print(f"      {text[:300]}...")
        print()

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python main.py ingest                              → Sube el PDF a Pinecone + FAISS local")
        print('  python main.py query "tu pregunta"                 → Busca (FAISS local → Pinecone fallback)')
        print('  python main.py query "tu pregunta" --engine faiss  → Forzar búsqueda en FAISS')
        print('  python main.py query "tu pregunta" --engine pinecone → Forzar búsqueda en Pinecone')
        print("  python main.py sync                                → Sincroniza Pinecone → FAISS local")
        sys.exit(0)

    command = sys.argv[1]

    if command == "ingest":
        ingest()
    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ Debes proporcionar una pregunta. Ejemplo:")
            print('   python main.py query "¿Qué saberes básicos hay en matemáticas?"')
            sys.exit(1)

        # Detectar flag --engine
        engine = "auto"
        args = sys.argv[2:]
        if "--engine" in args:
            idx = args.index("--engine")
            if idx + 1 < len(args):
                engine = args[idx + 1]
                if engine not in ("faiss", "pinecone", "auto"):
                    print(f"❌ Motor desconocido: {engine}. Usa 'faiss', 'pinecone' o 'auto'.")
                    sys.exit(1)
                args = args[:idx] + args[idx + 2:]
            else:
                print("❌ Debes especificar un motor después de --engine (faiss, pinecone, auto).")
                sys.exit(1)

        question = " ".join(args)
        query(question, engine=engine)
    elif command == "sync":
        sync()
    else:
        print(f"❌ Comando desconocido: {command}")
        print("   Usa 'ingest', 'query' o 'sync'")

