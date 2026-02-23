"""
Base de datos vectorial con Pinecone + HuggingFace (sentence-transformers).
Lee el PDF Saberes_2ESO.pdf, genera embeddings y los almacena en Pinecone.
Permite hacer consultas por similitud semántica.

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
import fitz  # PyMuPDF
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
MIN_CHUNK_CHARS = 60    # descartar chunks demasiado pequeños (ruido)
PDF_PATH = os.path.join(os.path.dirname(__file__), "Saberes_2ESO.pdf")


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
    Chunking semántico en 3 niveles:
      1. Detecta encabezados → marca inicio de nueva sección.
      2. Agrupa párrafos de la misma sección hasta MAX_CHUNK_CHARS.
      3. Si un párrafo solo ya supera MAX_CHUNK_CHARS, lo divide por oraciones.
    Cada chunk incluye: texto, página, sección (asignatura/trimestre).
    """
    chunks = []

    def flush_chunk(text: str, section: str, page: int):
        """Añade el chunk a la lista si supera el mínimo de caracteres."""
        text = text.strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append({"text": text, "section": section, "page": page})

    current_section = "General"
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
                    if len(current_chunk) + len(para) + 2 <= MAX_CHUNK_CHARS:
                        current_chunk = (current_chunk + "\n\n" + para).strip()
                        current_page = page_num
                    else:
                        flush_chunk(current_chunk, current_section, current_page)
                        if len(para) > MAX_CHUNK_CHARS:
                            for frag in split_by_sentences(para, MAX_CHUNK_CHARS):
                                flush_chunk(frag, current_section, page_num)
                            current_chunk = ""
                        else:
                            current_chunk = para
                        current_page = page_num
                continue

            if is_heading(stripped):
                # Guardar lo acumulado antes de cambiar de sección
                if paragraph_buffer.strip():
                    para = paragraph_buffer.strip()
                    paragraph_buffer = ""
                    current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
                flush_chunk(current_chunk, current_section, current_page)
                current_chunk = ""
                current_section = stripped
                current_page = page_num
            else:
                paragraph_buffer = (paragraph_buffer + " " + stripped).strip()

        # Al acabar la página, volcar el buffer restante
        if paragraph_buffer.strip():
            para = paragraph_buffer.strip()
            if len(current_chunk) + len(para) + 2 <= MAX_CHUNK_CHARS:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                flush_chunk(current_chunk, current_section, current_page)
                if len(para) > MAX_CHUNK_CHARS:
                    for frag in split_by_sentences(para, MAX_CHUNK_CHARS):
                        flush_chunk(frag, current_section, page_num)
                    current_chunk = ""
                else:
                    current_chunk = para
            current_page = page_num

    # Volcar el último chunk pendiente
    flush_chunk(current_chunk, current_section, current_page)

    return chunks


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
                "text":    chunks[j]["text"],
                "page":    chunks[j]["page"],
                "section": chunks[j]["section"],
            }
            batch.append((vector_id, embeddings[j].tolist(), metadata))
        index.upsert(vectors=batch)
        print(f"   → Subidos {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"\n✅ Ingesta completada: {len(chunks)} fragmentos en el índice '{INDEX_NAME}'.")


# ──────────────────────────────────────────────
# Consulta
# ──────────────────────────────────────────────
def query(question: str, top_k: int = 5):
    """Busca los fragmentos más relevantes para una pregunta."""
    api_key = load_env()

    model = SentenceTransformer(EMBEDDING_MODEL)
    q_embedding = model.encode(question).tolist()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    results = index.query(vector=q_embedding, top_k=top_k, include_metadata=True)

    print(f"\n🔎 Pregunta: {question}")
    print(f"📊 Top {top_k} resultados:\n")
    for i, match in enumerate(results["matches"], 1):
        score   = match["score"]
        text    = match["metadata"]["text"]
        page    = match["metadata"]["page"]
        section = match["metadata"].get("section", "—")
        print(f"  [{i}] (similitud: {score:.4f}) — {section} | Página {page}")
        print(f"      {text[:300]}...")
        print()

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python main.py ingest               → Sube el PDF a Pinecone")
        print('  python main.py query "tu pregunta"  → Busca en la base de datos')
        sys.exit(0)

    command = sys.argv[1]

    if command == "ingest":
        ingest()
    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ Debes proporcionar una pregunta. Ejemplo:")
            print('   python main.py query "¿Qué saberes básicos hay en matemáticas?"')
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        query(question)
    else:
        print(f"❌ Comando desconocido: {command}")
        print("   Usa 'ingest' o 'query'")

