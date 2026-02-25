# 📚 Base de Datos Vectorial — Saberes Básicos 2º ESO

Base de datos vectorial con **Pinecone**, **FAISS** y **HuggingFace** que almacena el contenido del archivo `Saberes_2ESO.pdf` y permite hacer consultas por similitud semántica. Cada fragmento se asocia a su **asignatura** y **trimestre** de origen.

---

## 🧰 Tecnologías

| Tecnología | Uso |
|---|---|
| [Pinecone](https://www.pinecone.io/) | Base de datos vectorial en la nube (serverless) |
| [FAISS](https://github.com/facebookresearch/faiss) | Búsqueda vectorial local ultrarrápida y offline |
| [sentence-transformers](https://www.sbert.net/) | Modelo de embeddings de HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, multilingüe) |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracción de texto del PDF |
| [Streamlit](https://streamlit.io/) | Interfaz web para consultas |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Gestión de variables de entorno |

---

## 📁 Estructura del proyecto

```
BBDD_vectorial_temas/
├── .env                 # Clave API de Pinecone (NO subir a Git)
├── .gitignore           # Excluye .env, .venv/, __pycache__/
├── .venv/               # Entorno virtual de Python
├── main.py              # Script principal (ingesta + consulta + sync)
├── app.py               # Interfaz web con Streamlit
├── test_chunks.py       # Script para probar el chunking y generar vista previa
├── chunks_preview.txt   # Vista previa de los chunks generados
├── requirements.txt     # Dependencias del proyecto
├── faiss_index/         # Índice FAISS local (generado automáticamente)
├── Saberes_2ESO.pdf     # PDF con los saberes básicos de 2º ESO
└── README.md            # Este archivo
```

---

## 🚀 Puesta en marcha

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd BBDD_vectorial_temas
```

### 2. Crear el entorno virtual e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar la clave de Pinecone

1. Crea una cuenta gratuita en [pinecone.io](https://www.pinecone.io/).
2. Ve a **API Keys** en el dashboard y copia tu clave.
3. Edita el archivo `.env` y pega tu clave:

```
PINECONE_API_KEY=pcsk_XXXXXXXXXXXXXXX
```

### 4. Ingestar el PDF en Pinecone + FAISS

```bash
.venv/bin/python3 main.py ingest
```

Esto hará lo siguiente:
1. **Lee** el PDF `Saberes_2ESO.pdf` con PyMuPDF.
2. **Divide** el texto en chunks semánticos por **asignatura** y **trimestre**: detecta encabezados de asignatura y marcas de trimestre (`1.º trimestre`, `2.º trimestre`, `3.º trimestre`) para crear un chunk independiente por cada combinación.
3. **Enriquece** cada chunk con un prefijo de contexto (`[Asignatura] [Trimestre]`) para mejorar la calidad del embedding.
4. **Genera embeddings** en local con el modelo `paraphrase-multilingual-MiniLM-L12-v2` de HuggingFace.
5. **Crea el índice** `saberes-2eso` en Pinecone (si no existe).
6. **Sube** todos los vectores con metadatos: texto, página, sección y trimestre.
7. **Guarda** un índice FAISS local para búsquedas offline.

Salida esperada:
```
📄 Leyendo PDF: /ruta/Saberes_2ESO.pdf
   → 3 páginas con texto.
✂️  Aplicando chunking semántico (max=800 chars por chunk)...
   → 29 fragmentos generados.
   → 10 secciones detectadas: Educación Física, Física y Química, ...
🤗 Cargando modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2
🔢 Generando embeddings para 29 fragmentos...
🌲 Conectando con Pinecone...
   ℹ️  El índice 'saberes-2eso' ya existe.
🗑️  Limpiando índice anterior...
🚀 Subiendo vectores a Pinecone...
   → Subidos 29/29
💾 Construyendo índice FAISS local...
✅ Ingesta completada: 29 fragmentos en el índice 'saberes-2eso' (Pinecone + FAISS local).
```

### 5. Hacer consultas (CLI)

```bash
.venv/bin/python3 main.py query "¿Qué saberes básicos hay en matemáticas en el primer trimestre?"
```

Devuelve los 5 fragmentos más relevantes ordenados por similitud:

```
🔎 Pregunta: ¿Qué saberes básicos hay en matemáticas en el primer trimestre?
⚡ Motor: FAISS (local)
📊 Top 5 resultados:

  [1] (similitud: 0.7523) — Matemáticas | 1.º trimestre | Página 1
      [Matemáticas] [1.º trimestre] Números, operaciones, proporciones y geometría básica...
```

Se puede forzar el motor de búsqueda con `--engine`:

```bash
.venv/bin/python3 main.py query "pregunta" --engine faiss     # Solo FAISS (local)
.venv/bin/python3 main.py query "pregunta" --engine pinecone  # Solo Pinecone (cloud)
```

### 6. Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Se abrirá una interfaz en el navegador con:
- Campo de búsqueda semántica
- Filtro por similitud mínima
- Resultados con sección, trimestre y página
- Barra de similitud visual con colores

### 7. Sincronizar Pinecone → FAISS

```bash
.venv/bin/python3 main.py sync
```

Descarga los vectores de Pinecone y reconstruye el índice FAISS local. Útil cuando se actualiza Pinecone desde otro entorno.

---

## ⚙️ Cómo funciona

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Saberes_2ESO │────▶│  PyMuPDF extrae   │────▶│  Texto plano │
│    .pdf      │     │  texto por página  │     │  por página  │
└──────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
                                         ┌─────────────────────────┐
                                         │  Chunking semántico      │
                                         │  1. Detecta asignaturas  │
                                         │  2. Detecta trimestres   │
                                         │  3. Agrupa párrafos      │
                                         │  4. Divide por oraciones │
                                         │  → 1 chunk/asig./trim.   │
                                         └──────────┬──────────────┘
                                                    │
                                                    ▼
                                         ┌───────────────────────┐
                                         │ sentence-transformers  │
                                         │ genera embeddings      │
                                         │ (384 dimensiones)      │
                                         └──────┬────────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                             ┌─────────────┐        ┌─────────────┐
                             │  INGESTA:   │        │  CONSULTA:  │
                             │  Pinecone + │        │  FAISS local│
                             │  FAISS local│        │  → Pinecone │
                             └─────────────┘        │  (fallback) │
                                                    └─────────────┘
```

1. **Extracción**: PyMuPDF lee el PDF y extrae el texto de cada página.
2. **Chunking semántico (4 niveles)**:
   - **Nivel 1 — Secciones**: detecta los nombres de asignaturas como encabezados y los usa para separar el contenido.
   - **Nivel 2 — Trimestres**: detecta marcas de trimestre (`1.º trimestre`, `2.º trimestre`, `3.º trimestre`) y crea un chunk separado por cada uno, conservando la sección a la que pertenece.
   - **Nivel 3 — Párrafos**: dentro de cada trimestre/sección, agrupa párrafos juntos hasta el límite de 800 caracteres.
   - **Nivel 4 — Oraciones**: si un párrafo supera el límite, se divide por oraciones completas (nunca a mitad de frase).
3. **Enriquecimiento**: cada chunk se prefija con `[Asignatura] [Trimestre]` para que el embedding capture el contexto completo.
4. **Embeddings**: el modelo `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) convierte cada chunk en un vector de 384 dimensiones. Es multilingüe (50+ idiomas, incluido español) y se ejecuta **en local**, sin llamadas a APIs externas.
5. **Almacenamiento dual**:
   - **Pinecone**: índice serverless (AWS, us-east-1, métrica coseno) con metadatos de texto, página, sección y trimestre.
   - **FAISS**: índice local para búsquedas rápidas y offline.
6. **Consulta**: la pregunta del usuario se convierte en un embedding con el mismo modelo. Se busca primero en FAISS (rápido, offline) y si no está disponible, se usa Pinecone como fallback.

---

## 📋 Comandos disponibles

| Comando | Descripción |
|---|---|
| `python main.py ingest` | Sube el contenido del PDF a Pinecone + FAISS local |
| `python main.py query "pregunta"` | Busca fragmentos relevantes (FAISS → Pinecone fallback) |
| `python main.py query "pregunta" --engine faiss` | Forzar búsqueda solo en FAISS |
| `python main.py query "pregunta" --engine pinecone` | Forzar búsqueda solo en Pinecone |
| `python main.py sync` | Sincroniza Pinecone → FAISS local |
| `streamlit run app.py` | Abre la interfaz web de búsqueda |
| `python test_chunks.py` | Genera `chunks_preview.txt` con la vista previa del chunking |

---

## 📌 Notas

- La primera ejecución descargará el modelo `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB). Las siguientes usan la caché local.
- Pinecone tier gratuito permite 1 índice serverless con hasta 2 GB de almacenamiento.
- Si quieres reingestar el PDF, puedes volver a ejecutar `ingest`; los vectores se sobreescriben.
- Cada chunk incluye metadatos de **sección** (asignatura) y **trimestre**, lo que permite búsquedas contextuales precisas como *"¿Qué se estudia en música en el 2.º trimestre?"*.

