"""
Interfaz Streamlit — Buscador de Saberes Básicos 2º ESO.

Flujo idéntico al de main.py:
  1. EXTRACCIÓN  — Qwen2.5-0.5B-Instruct (HF Router · featherless-ai) infiere
                   asignatura y trimestre de la query con el prompt estructurado.
  2. BÚSQUEDA    — La query completa se vectoriza (SentenceTransformer) y se busca
                   en FAISS. Los filtros extraídos se aplican sobre los metadatos.

Ejecutar:
    streamlit run app.py
"""

import os
import re
import json
import unicodedata

import numpy as np
import faiss
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login as hf_login

# ──────────────────────────────────────────────
# Configuración (sincronizada con main.py)
# ──────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM       = 384
FAISS_DIR           = os.path.join(BASE_DIR, "faiss_index")
FAISS_INDEX_PATH    = os.path.join(FAISS_DIR, "index.faiss")
FAISS_METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

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
# Carga de entorno y recursos (cacheados)
# ──────────────────────────────────────────────
@st.cache_resource
def _init_env() -> str | None:
    """Carga .env, autentica en HF Hub y devuelve HF_TOKEN."""
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False)
    return hf_token


@st.cache_resource
def _load_embedding_model() -> SentenceTransformer:
    """Carga el modelo de embeddings una sola vez."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def _load_faiss():
    """Carga el índice FAISS y los metadatos desde disco."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        return None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ──────────────────────────────────────────────
# Extractor de filtros (HF Router)
# ──────────────────────────────────────────────
def extract_query_filters(query_text: str, hf_token: str | None) -> dict:
    """
    Envía la query al modelo Qwen2.5-0.5B-Instruct via HF Router (featherless-ai).
    Retorna {"subject": str|None, "trimester": str|None}.
    """
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "model": QA_EXTRACTOR_MODEL,
        "messages": [{"role": "user",
                      "content": QA_EXTRACTOR_PROMPT.replace("{query}", query_text)}],
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
    except Exception:
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


def _normalize(value: str) -> str:
    """Minúsculas → sin puntuación → sin tildes → ordinales a dígito."""
    v = _PUNCT_RE.sub(" ", value.lower().strip())
    v = "".join(c for c in unicodedata.normalize("NFD", v) if unicodedata.category(c) != "Mn")
    return " ".join(_ORDINAL_MAP.get(t, t) for t in v.split())


def _matches(extracted: str, stored: str) -> bool:
    """True si los valores normalizados se contienen mutuamente."""
    ne, ns = _normalize(extracted), _normalize(stored)
    return ne in ns or ns in ne


# ──────────────────────────────────────────────
# Búsqueda FAISS con filtros
# ──────────────────────────────────────────────
def search_faiss(question: str, filters: dict, top_k: int) -> list[dict] | None:
    """
    Vectoriza la query original completa, busca en FAISS y filtra por metadatos.
    Retorna lista de resultados o None si no existe el índice.
    """
    index, metadata = _load_faiss()
    if index is None:
        return None

    model = _load_embedding_model()
    q_vec = np.array([model.encode(question)], dtype="float32")
    faiss.normalize_L2(q_vec)

    active = {k: v for k, v in filters.items() if v is not None}
    pool   = min(top_k * 5 if active else top_k, index.ntotal)
    scores, indices = index.search(q_vec, pool)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta      = metadata[idx]
        section   = meta.get("section", "")
        trimester = meta.get("trimester", "General")

        if active:
            if active.get("subject")   and not _matches(active["subject"],   section):
                continue
            if active.get("trimester") and not _matches(active["trimester"], trimester):
                continue

        results.append({
            "score":    float(score),
            "text":     meta["text"],
            "page":     meta["page"],
            "section":  section,
            "trimester": trimester,
        })
        if len(results) >= top_k:
            break
    return results


# ──────────────────────────────────────────────
# Interfaz Streamlit
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Saberes Básicos 2º ESO",
    page_icon="📚",
    layout="centered",
)

# Inicializar entorno (HF_TOKEN + HF login)
hf_token = _init_env()

st.title("📚 Buscador de Saberes Básicos — 2º ESO")
st.caption("Búsqueda semántica con extracción automática de asignatura y trimestre · FAISS + HuggingFace")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Opciones")
    top_k      = st.slider("Número de resultados", 1, 10, 5)
    min_score  = st.slider("Similitud mínima", 0.0, 1.0, 0.0, 0.05)
    show_score = st.checkbox("Mostrar puntuación de similitud", value=True)

    st.divider()
    st.markdown("**Ejemplos de búsqueda:**")
    examples = [
        "¿Qué saberes básicos hay en matemáticas en el primer trimestre?",
        "¿Qué contenidos entran en inglés en el segundo trimestre?",
        "Dime los criterios de evaluación de Lengua Castellana.",
        "¿Cuáles son los objetivos del tercer trimestre de Ciencias Naturales?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# ── Comprobación del índice FAISS ──────────────────────────────────────────
index, _ = _load_faiss()
if index is None:
    st.error("⚠️  No se encontró el índice FAISS. Ejecuta primero `python main.py ingest`.")
    st.stop()

# ── Campo de búsqueda ──────────────────────────────────────────────────────
question = st.text_input(
    "🔎 Escribe tu pregunta:",
    value=st.session_state.get("question", ""),
    placeholder="Ej: ¿Qué se estudia en matemáticas en el primer trimestre?",
)

if not question:
    st.info("Escribe una pregunta o selecciona un ejemplo del panel lateral.")
    st.stop()

# ── Paso 1: extracción de filtros ──────────────────────────────────────────
with st.spinner(f"🧠 Extrayendo filtros con {QA_EXTRACTOR_MODEL}..."):
    if not hf_token:
        st.warning("⚠️  HF_TOKEN no encontrado en .env — búsqueda sin filtros de extracción.")
        filters = {"subject": None, "trimester": None}
    else:
        filters = extract_query_filters(question, hf_token)

subject_raw   = filters.get("subject")
trimester_raw = filters.get("trimester")

col1, col2 = st.columns(2)
with col1:
    st.info(f"📚 Asignatura detectada: **{subject_raw or 'ninguna'}**")
with col2:
    st.info(f"📅 Trimestre detectado: **{trimester_raw or 'ninguno'}**")

# ── Paso 2: búsqueda vectorial con query original + filtros ────────────────
with st.spinner("🔍 Buscando en la base de datos vectorial..."):
    results = search_faiss(question, filters, top_k)

if results is None:
    st.error("Error al buscar. Verifica que el índice FAISS esté generado.")
    st.stop()

filtered = [r for r in results if r["score"] >= min_score]

if not filtered:
    st.warning("No se encontraron resultados con la similitud mínima seleccionada.")
    st.stop()

st.markdown(f"### 📊 {len(filtered)} resultado(s) encontrado(s)")

for i, result in enumerate(filtered, 1):
    score     = result["score"]
    section   = result["section"]
    trimester = result["trimester"]
    page      = result["page"]
    text      = result["text"]

    color = "🟢" if score >= 0.5 else ("🟡" if score >= 0.3 else "🔴")
    header = f"{color} **{section}** — {trimester} — Página {page}"
    if show_score:
        header += f" &nbsp;·&nbsp; Similitud: `{score:.4f}`"

    with st.expander(header, expanded=(i <= 3)):
        st.markdown(text)
        st.progress(min(score, 1.0))

