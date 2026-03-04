"""
Base de datos vectorial — Saberes Básicos 2º ESO.
Pinecone (almacenamiento) + FAISS (búsqueda local) + HuggingFace (embeddings + extracción).

Flujo de consulta:
  1. EXTRACCIÓN  — Qwen2.5-0.5B-Instruct (HF Router) analiza la query con un prompt
                   estructurado e infiere 'asignatura' y 'trimestre' del texto.
  2. BÚSQUEDA    — La query completa se convierte en embedding y se busca en FAISS/Pinecone.
                   Los valores extraídos se aplican como filtros sobre los metadatos.

Chunking semántico:
  - El campo 'text' contiene únicamente el criterio (texto limpio).
  - La asignatura (section) y el trimestre se almacenan solo como metadatos.
"""

import os
import re
import sys
import time
import json
import unicodedata
import fitz
import numpy as np
import faiss
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from huggingface_hub import login as hf_login
import requests

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
INDEX_NAME     = "saberes-2eso"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM  = 384
MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 20
PDF_PATH           = os.path.join(os.path.dirname(__file__), "Saberes_2ESO.pdf")
FAISS_DIR          = os.path.join(os.path.dirname(__file__), "faiss_index")
FAISS_INDEX_PATH   = os.path.join(FAISS_DIR, "index.faiss")
FAISS_METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

# ── Extractor de filtros (HF Router · featherless-ai) ──────────────────────
QA_EXTRACTOR_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
QA_EXTRACTOR_API_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"
QA_EXTRACTOR_PROMPT  = (
    "Actúa como un extractor de datos especializado en currículo educativo. "
    "Tu tarea es analizar la consulta (query) del usuario y extraer dos campos "
    "específicos: trimestre y asignatura.\n"
    "Reglas de extracción:\n"
    "- trimestre: Debe seguir el formato \"X.º trimestre\" (ej. 1.º trimestre, 2.º trimestre).\n"
    "- asignatura: El nombre de la materia (ej. inglés, matemáticas, historia).\n"
    "Valores ausentes: Si alguno de estos datos no se menciona explícitamente en la "
    "consulta, asigna el valor null.\n"
    "Formato de salida: Responde únicamente con un objeto JSON.\n"
    "Ejemplos:\n"
    "Entrada: \"¿Qué entra en el proyecto oral de inglés del primer trimestre?\"\n"
    "Salida: {\"trimestre\": \"1.º trimestre\", \"asignatura\": \"inglés\"}\n"
    "Entrada: \"Dime los criterios de evaluación de matemáticas.\"\n"
    "Salida: {\"trimestre\": null, \"asignatura\": \"matemáticas\"}\n"
    "Ahora extrae los campos de la siguiente consulta:\n"
    "Entrada: \"{query}\"\n"
    "Salida:"
)


# ──────────────────────────────────────────────
# Variables de entorno
# ──────────────────────────────────────────────
def load_env() -> str:
    """Carga .env, autentica en HF Hub y devuelve la API key de Pinecone."""
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key.startswith("tu_clave"):
        print("❌ ERROR: Falta la PINECONE_API_KEY en el archivo .env")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False)
    else:
        print("⚠️  HF_TOKEN no encontrado en .env — se usará acceso anónimo al Hub.")

    return api_key


# ──────────────────────────────────────────────
# Extracción de texto del PDF
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Chunking semántico
# ──────────────────────────────────────────────
def is_heading(line: str) -> bool:
    """Detecta encabezados de asignatura o título principal del documento."""
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
    return any(re.match(p, line, re.IGNORECASE) for p in patterns)


def split_by_trimesters(text: str) -> list[tuple[str, str]]:
    """
    Divide un texto en fragmentos por marcas de trimestre inline.
    Retorna lista de tuplas (nombre_trimestre, contenido).
    Si no hay marcas, retorna [("General", texto)].
    """
    parts = re.split(r"(\d\.?[°ºo]?\s*trimestre)\s*", text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        return [("General", text.strip())]

    result = []
    if parts[0].strip():
        result.append(("General", parts[0].strip()))
    for i in range(1, len(parts), 2):
        m = re.match(r"(\d)", parts[i].strip())
        tri = f"{m.group(1)}.º trimestre" if m else parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result.append((tri, content))
    return result


def split_by_sentences(text: str, max_chars: int) -> list[str]:
    """Divide texto largo en fragmentos respetando oraciones completas."""
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
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


def split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Chunking semántico: detecta secciones y trimestres, agrupa párrafos
    hasta MAX_CHUNK_CHARS y divide por oraciones si supera el límite.
    El campo 'text' contiene únicamente el criterio.
    La asignatura y el trimestre se almacenan exclusivamente como metadatos.
    """
    chunks = []

    def flush(text: str, section: str, trimester: str, page: int):
        text = text.strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append({"text": text, "section": section,
                           "trimester": trimester, "page": page})

    def process(para: str, section: str, trimester: str, page: int,
                acc: str) -> tuple[str, str]:
        tri_parts = split_by_trimesters(para)
        if len(tri_parts) == 1 and tri_parts[0][0] == "General":
            text = tri_parts[0][1]
            if len(acc) + len(text) + 2 <= MAX_CHUNK_CHARS:
                return (acc + "\n\n" + text).strip(), trimester
            flush(acc, section, trimester, page)
            if len(text) > MAX_CHUNK_CHARS:
                for frag in split_by_sentences(text, MAX_CHUNK_CHARS):
                    flush(frag, section, trimester, page)
                return "", trimester
            return text, trimester

        flush(acc, section, trimester, page)
        last_tri = trimester
        for tri_name, tri_content in tri_parts:
            if tri_name != "General":
                last_tri = tri_name
            if tri_content:
                if len(tri_content) > MAX_CHUNK_CHARS:
                    for frag in split_by_sentences(tri_content, MAX_CHUNK_CHARS):
                        flush(frag, section, last_tri, page)
                else:
                    flush(tri_content, section, last_tri, page)
        return "", last_tri

    cur_section, cur_tri, cur_acc, cur_page = "General", "General", "", 1

    for page_data in pages:
        page_num = page_data["page"]
        buf = ""
        for line in page_data["text"].split("\n"):
            s = line.strip()
            if not s:
                if buf:
                    cur_acc, cur_tri = process(buf, cur_section, cur_tri, page_num, cur_acc)
                    cur_page = page_num
                    buf = ""
                continue
            if is_heading(s):
                if buf:
                    cur_acc, cur_tri = process(buf, cur_section, cur_tri, page_num, cur_acc)
                    buf = ""
                flush(cur_acc, cur_section, cur_tri, cur_page)
                cur_acc, cur_section, cur_tri, cur_page = "", s, "General", page_num
            else:
                buf = (buf + " " + s).strip()
        if buf:
            cur_acc, cur_tri = process(buf, cur_section, cur_tri, page_num, cur_acc)
            cur_page = page_num

    flush(cur_acc, cur_section, cur_tri, cur_page)
    return chunks


# ──────────────────────────────────────────────
# FAISS — guardar / cargar
# ──────────────────────────────────────────────
def save_faiss_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    """Construye el índice FAISS (coseno) y guarda índice + metadatos en disco."""
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)

    metadata = [
        {"id": f"chunk-{i}", "text": c["text"], "page": c["page"],
         "section": c["section"], "trimester": c.get("trimester", "General")}
        for i, c in enumerate(chunks)
    ]
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   💾 Índice FAISS guardado en {FAISS_DIR}/ ({index.ntotal} vectores)")


def load_faiss_index():
    """Carga índice FAISS y metadatos desde disco. Retorna (None, None) si no existen."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        return None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ──────────────────────────────────────────────
# Extractor de filtros (HF Router)
# ──────────────────────────────────────────────
def extract_query_filters(query_text: str) -> dict:
    """
    Envía la query al modelo Qwen2.5-0.5B-Instruct (HF Router · featherless-ai)
    con el prompt estructurado QA_EXTRACTOR_PROMPT.
    Retorna {"subject": str|None, "trimester": str|None}.
    """
    hf_token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "model": QA_EXTRACTOR_MODEL,
        "messages": [{"role": "user", "content": QA_EXTRACTOR_PROMPT.replace("{query}", query_text)}],
        "max_tokens": 80,
        "temperature": 0,
        "stream": False,
    }
    filters: dict[str, str | None] = {"subject": None, "trimester": None}
    try:
        r = requests.post(QA_EXTRACTOR_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{[^{}]*}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            asig = data.get("asignatura")
            if asig and str(asig).lower() not in ("null", "none", ""):
                filters["subject"] = str(asig).strip()
            tri = data.get("trimestre")
            if tri and str(tri).lower() not in ("null", "none", ""):
                filters["trimester"] = str(tri).strip()
    except requests.exceptions.Timeout:
        print("   ⚠️  Timeout al llamar al HF Router → búsqueda sin filtros.")
    except requests.exceptions.HTTPError as e:
        print(f"   ⚠️  Error HTTP del HF Router ({e}) → búsqueda sin filtros.")
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return filters


# ──────────────────────────────────────────────
# Normalización para comparación de filtros
# ──────────────────────────────────────────────
_ORDINAL_MAP = {
    "primer": "1", "primero": "1", "primera": "1", "1.º": "1", "1º": "1", "1er": "1",
    "segundo": "2", "segunda": "2", "2.º": "2", "2º": "2",
    "tercer": "3", "tercero": "3", "tercera": "3", "3.º": "3", "3º": "3",
}
_PUNCT_RE = re.compile(r"[¿?¡!.,;:()\[\]\"']+")


def _normalize_filter(value: str) -> str:
    """Minúsculas → sin puntuación → sin tildes → ordinales a dígito."""
    v = _PUNCT_RE.sub(" ", value.lower().strip())
    v = "".join(c for c in unicodedata.normalize("NFD", v) if unicodedata.category(c) != "Mn")
    return " ".join(_ORDINAL_MAP.get(t, t) for t in v.split())


def _filters_match(extracted: str, stored: str) -> bool:
    """True si los valores normalizados se contienen mutuamente (subcadena)."""
    ne, ns = _normalize_filter(extracted), _normalize_filter(stored)
    return ne in ns or ns in ne


# ──────────────────────────────────────────────
# Búsqueda FAISS con filtros
# ──────────────────────────────────────────────
def query_faiss(question: str, top_k: int = 5, filters: dict | None = None):
    """
    Busca en FAISS con la query original y aplica filtros de metadatos.
    Retorna lista de resultados o None si no existe el índice.
    """
    index, metadata = load_faiss_index()
    if index is None:
        return None

    model = SentenceTransformer(EMBEDDING_MODEL)
    q_vec = np.array([model.encode(question)], dtype="float32")
    faiss.normalize_L2(q_vec)

    active = {k: v for k, v in (filters or {}).items() if v is not None}
    pool   = min(top_k * 5 if active else top_k, index.ntotal)
    scores, indices = index.search(q_vec, pool)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta     = metadata[idx]
        section  = meta.get("section", "")
        trimester = meta.get("trimester", "General")

        if active:
            if active.get("subject")   and not _filters_match(active["subject"],   section):
                continue
            if active.get("trimester") and not _filters_match(active["trimester"], trimester):
                continue

        results.append({
            "score": float(score),
            "id": meta["id"],
            "metadata": {"text": meta["text"], "page": meta["page"],
                         "section": section, "trimester": trimester},
        })
        if len(results) >= top_k:
            break
    return results


# ──────────────────────────────────────────────
# Pinecone — utilidades
# ──────────────────────────────────────────────
def create_or_get_index(pc: Pinecone) -> None:
    """Crea el índice Pinecone si no existe y espera a que esté listo."""
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        print(f"   → Creando índice '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("   ⏳ Esperando...")
            time.sleep(2)
        print("   ✅ Índice creado.")
    else:
        print(f"   ℹ️  El índice '{INDEX_NAME}' ya existe.")


def sync():
    """Descarga vectores de Pinecone y reconstruye el índice FAISS local."""
    api_key = load_env()
    print("🌲 Conectando con Pinecone...")
    pc    = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        print("⚠️  El índice de Pinecone está vacío. Ejecuta 'ingest' primero.")
        return
    print(f"   → {stats.total_vector_count} vectores en Pinecone.")

    print("📥 Descargando vectores...")
    all_ids = []
    for batch in index.list():
        all_ids.extend(batch)

    fetched     = index.fetch(ids=all_ids).vectors
    sorted_ids  = sorted(fetched.keys(), key=lambda x: int(x.split("-")[1]))
    embeddings, chunks = [], []
    for vid in sorted_ids:
        v = fetched[vid]
        embeddings.append(v.values)
        chunks.append({
            "text": v.metadata.get("text", ""),
            "page": v.metadata.get("page", 0),
            "section": v.metadata.get("section", ""),
            "trimester": v.metadata.get("trimester", "General"),
        })

    save_faiss_index(np.array(embeddings, dtype="float32"), chunks)
    print(f"\n✅ Sincronización completada: {len(chunks)} vectores en FAISS.")


# ──────────────────────────────────────────────
# Ingesta
# ──────────────────────────────────────────────
def ingest():
    """Lee el PDF, genera embeddings y los sube a Pinecone + FAISS."""
    api_key = load_env()

    print(f"📄 Leyendo PDF: {PDF_PATH}")
    pages = extract_text_from_pdf(PDF_PATH)
    print(f"   → {len(pages)} páginas con texto.")

    print(f"✂️  Chunking semántico (max={MAX_CHUNK_CHARS} chars)...")
    chunks = split_into_chunks(pages)
    sections = sorted(set(c["section"] for c in chunks))
    print(f"   → {len(chunks)} fragmentos | {len(sections)} secciones: {', '.join(sections)}")

    print(f"🤗 Generando embeddings ({EMBEDDING_MODEL})...")
    model      = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True, batch_size=64)

    print("🌲 Conectando con Pinecone...")
    pc = Pinecone(api_key=api_key)
    create_or_get_index(pc)
    pine_index = pc.Index(INDEX_NAME)

    print("🗑️  Limpiando índice anterior...")
    pine_index.delete(delete_all=True)

    print("🚀 Subiendo vectores a Pinecone...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = [
            (f"chunk-{j}", embeddings[j].tolist(),
             {"text": chunks[j]["text"], "page": chunks[j]["page"],
              "section": chunks[j]["section"], "trimester": chunks[j].get("trimester", "General")})
            for j in range(i, min(i + batch_size, len(chunks)))
        ]
        pine_index.upsert(vectors=batch)
        print(f"   → {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("💾 Construyendo índice FAISS local...")
    save_faiss_index(embeddings, chunks)
    print(f"\n✅ Ingesta completada: {len(chunks)} fragmentos en '{INDEX_NAME}'.")


# ──────────────────────────────────────────────
# Consulta principal
# ──────────────────────────────────────────────
def query(question: str, top_k: int = 5, engine: str = "auto"):
    """
    Busca fragmentos relevantes para la pregunta del usuario.

    Flujo:
      1. EXTRACCIÓN  — Qwen2.5-0.5B-Instruct extrae asignatura y trimestre de la query.
      2. BÚSQUEDA    — La query completa original se vectoriza y se busca en FAISS/Pinecone.
                       Los filtros extraídos se aplican sobre los metadatos.
    """
    print(f"\n🔎 Pregunta: {question}")
    api_key = load_env()

    # ── Paso 1: extracción de filtros ──────────────────────────────────────
    print(f"🧠 Extrayendo filtros (HF Router · {QA_EXTRACTOR_MODEL})...")
    filters = extract_query_filters(question)

    subject_raw   = filters.get("subject")
    trimester_raw = filters.get("trimester")

    if subject_raw or trimester_raw:
        norm_s = _normalize_filter(subject_raw)   if subject_raw   else None
        norm_t = _normalize_filter(trimester_raw) if trimester_raw else None
        print(f"   🏷️  Extraídos   → asignatura: {subject_raw or '—'} | trimestre: {trimester_raw or '—'}")
        print(f"   🔧 Normalizados → asignatura: {norm_s or '—'} | trimestre: {norm_t or '—'}")
    else:
        norm_s = norm_t = None
        print("   ℹ️  Sin filtros detectados → búsqueda global.")

    # ── Paso 2: búsqueda vectorial con la query original + filtros ──────────
    has_filters = bool(subject_raw or trimester_raw)
    print("🔍 Búsqueda vectorial" + (" + filtros estructurados" if has_filters else "") + "...")

    if engine in ("faiss", "auto"):
        results = query_faiss(question, top_k, filters=filters)
        if results is not None:
            print("⚡ Motor: FAISS (local)")
            if not results:
                print("   ⚠️  Sin resultados. Prueba una búsqueda más amplia.")
                return results
            print(f"📊 Top {len(results)} resultados:\n")
            for i, r in enumerate(results, 1):
                m = r["metadata"]
                print(f"  [{i}] (similitud: {r['score']:.4f}) — {m['section']} | {m['trimester']} | Pág. {m['page']}")
                print(f"      {m['text'][:300]}...\n")
            return results
        if engine == "faiss":
            print("❌ Índice FAISS no encontrado. Ejecuta 'ingest' o 'sync' primero.")
            sys.exit(1)
        print("⚠️  Sin índice FAISS local → usando Pinecone...")

    # ── Pinecone ───────────────────────────────────────────────────────────
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    q_embedding = embed_model.encode(question).tolist()

    pine_index = Pinecone(api_key=api_key).Index(INDEX_NAME)
    meta_filter: dict = {}
    if norm_s:
        meta_filter["section"]   = {"$eq": norm_s}
    if norm_t:
        meta_filter["trimester"] = {"$eq": norm_t}

    kwargs: dict = dict(vector=q_embedding, top_k=top_k, include_metadata=True)
    if meta_filter:
        kwargs["filter"] = meta_filter

    results = pine_index.query(**kwargs)
    print("🌲 Motor: Pinecone (cloud)")
    matches = results.get("matches", [])
    if not matches:
        print("   ⚠️  Sin resultados. Prueba una búsqueda más amplia.")
        return results
    print(f"📊 Top {len(matches)} resultados:\n")
    for i, r in enumerate(matches, 1):
        m = r["metadata"]
        print(f"  [{i}] (similitud: {r['score']:.4f}) — {m.get('section','—')} | {m.get('trimester','—')} | Pág. {m['page']}")
        print(f"      {m['text'][:300]}...\n")
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python main.py ingest                               → Indexa el PDF en Pinecone + FAISS")
        print('  python main.py query "pregunta"                     → Busca (FAISS → Pinecone fallback)')
        print('  python main.py query "pregunta" --engine faiss      → Forzar FAISS')
        print('  python main.py query "pregunta" --engine pinecone   → Forzar Pinecone')
        print("  python main.py sync                                 → Sincroniza Pinecone → FAISS")
        sys.exit(0)

    command = sys.argv[1]

    if command == "ingest":
        ingest()

    elif command == "query":
        if len(sys.argv) < 3:
            print('❌ Falta la pregunta. Ejemplo: python main.py query "¿Qué saberes hay en matemáticas?"')
            sys.exit(1)
        engine = "auto"
        args   = sys.argv[2:]
        if "--engine" in args:
            idx = args.index("--engine")
            if idx + 1 < len(args):
                engine = args[idx + 1]
                if engine not in ("faiss", "pinecone", "auto"):
                    print(f"❌ Motor desconocido: {engine}. Usa 'faiss', 'pinecone' o 'auto'.")
                    sys.exit(1)
                args = args[:idx] + args[idx + 2:]
            else:
                print("❌ Especifica un motor: --engine faiss | pinecone | auto")
                sys.exit(1)
        query(" ".join(args), engine=engine)

    elif command == "sync":
        sync()

    else:
        print(f"❌ Comando desconocido: '{command}'. Usa 'ingest', 'query' o 'sync'.")
