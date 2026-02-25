"""
Interfaz web con Streamlit para consultar la base de datos vectorial.
Usa FAISS como motor de búsqueda local con fallback a Pinecone.

Ejecutar:
  streamlit run app.py
"""

import os
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# Configuración (misma que main.py)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
FAISS_METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")


# ──────────────────────────────────────────────
# Funciones de búsqueda
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Carga el modelo de embeddings (cacheado para no recargarlo)."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_faiss():
    """Carga el índice FAISS y los metadatos desde disco."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        return None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(question: str, top_k: int = 5):
    """Busca en FAISS los fragmentos más similares a la pregunta."""
    index, metadata = load_faiss()
    if index is None:
        return None

    model = load_model()
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
            "text": meta["text"],
            "page": meta["page"],
            "section": meta["section"],
        })
    return results


# ──────────────────────────────────────────────
# Interfaz Streamlit
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="📚 Saberes 2º ESO — Buscador",
    page_icon="🔍",
    layout="wide",
)

st.title("📚 Saberes 2º ESO")
st.caption("Buscador semántico con FAISS + Sentence Transformers")

# Verificar que el índice FAISS existe
index, metadata = load_faiss()
if index is None:
    st.error(
        "⚠️ No se encontró el índice FAISS. "
        "Ejecuta primero `python main.py ingest` para generar los datos."
    )
    st.stop()

st.success(f"✅ Índice FAISS cargado: **{index.ntotal}** fragmentos indexados")

# ── Barra lateral ──
with st.sidebar:
    st.header("⚙️ Opciones")
    top_k = st.slider("Número de resultados", min_value=1, max_value=20, value=5)
    show_score = st.checkbox("Mostrar puntuación de similitud", value=True)
    min_score = st.slider(
        "Similitud mínima",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filtra resultados con similitud menor a este valor",
    )
    st.divider()
    st.markdown("**Ejemplos de preguntas:**")
    example_questions = [
        "¿Qué se estudia en matemáticas?",
        "¿Qué temas hay en ciencias naturales?",
        "¿Qué se aprende en lengua castellana?",
        "¿Qué se hace en educación física?",
        "¿Qué se estudia en el primer trimestre?",
    ]
    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state["question"] = eq

# ── Campo de búsqueda ──
question = st.text_input(
    "🔎 Escribe tu pregunta:",
    value=st.session_state.get("question", ""),
    placeholder="Ej: ¿Qué se estudia en matemáticas en el primer trimestre?",
)

if question:
    with st.spinner("Buscando..."):
        results = search(question, top_k=top_k)

    if results is None:
        st.error("Error al buscar. Verifica que el índice FAISS esté generado.")
    else:
        # Filtrar por similitud mínima
        filtered = [r for r in results if r["score"] >= min_score]

        if not filtered:
            st.warning("No se encontraron resultados con la similitud mínima seleccionada.")
        else:
            st.markdown(f"### 📊 {len(filtered)} resultado(s) encontrado(s)")

            for i, result in enumerate(filtered, 1):
                score = result["score"]
                section = result["section"]
                page = result["page"]
                text = result["text"]

                # Barra de similitud con color
                if score >= 0.5:
                    color = "🟢"
                elif score >= 0.3:
                    color = "🟡"
                else:
                    color = "🔴"

                header = f"{color} **{section}** — Página {page}"
                if show_score:
                    header += f" — Similitud: `{score:.4f}`"

                with st.expander(header, expanded=(i <= 3)):
                    st.markdown(text)
                    st.progress(min(score, 1.0))

