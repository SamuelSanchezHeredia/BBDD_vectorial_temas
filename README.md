# 📚 Base de Datos Vectorial — Saberes Básicos 2º ESO

Base de datos vectorial con **Pinecone**, **FAISS** y **HuggingFace** que almacena el contenido del archivo `Saberes_2ESO.pdf` y permite hacer consultas por similitud semántica. Cada fragmento se asocia a su **asignatura** y **trimestre** de origen mediante metadatos independientes.

---

## 🧰 Tecnologías

| Tecnología | Uso |
|---|---|
| [Pinecone](https://www.pinecone.io/) | Base de datos vectorial en la nube (serverless) |
| [FAISS](https://github.com/facebookresearch/faiss) | Búsqueda vectorial local ultrarrápida y offline |
| [sentence-transformers](https://www.sbert.net/) | Modelo de embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, multilingüe) |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Extractor de filtros vía **HF Router featherless-ai** (API OpenAI-compatible): infiere asignatura y trimestre de la query mediante prompt estructurado — **sin ejecución local de PyTorch** |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracción de texto del PDF |
| [Streamlit](https://streamlit.io/) | Interfaz web para consultas (mismo flujo que `main.py`) |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Gestión de variables de entorno |
| [requests](https://docs.python-requests.org/) | Llamadas HTTP al HF Router para la extracción de filtros |

---

## 📁 Estructura del proyecto

```
BBDD_vectorial_temas/
├── .env                 # Claves API de Pinecone y HuggingFace (NO subir a Git)
├── .gitignore           # Excluye .env, .venv/, __pycache__/
├── .venv/               # Entorno virtual de Python
├── main.py              # Script principal (ingesta + consulta CLI + sync)
├── app.py               # Interfaz web con Streamlit (flujo idéntico a main.py)
├── test_chunks.py       # Script para probar el chunking y generar vista previa
├── chunks_preview.txt   # Vista previa de los chunks generados
├── requirements.txt     # Dependencias del proyecto
├── faiss_index/         # Índice FAISS local (generado automáticamente)
│   ├── index.faiss
│   └── metadata.json
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

### 3. Configurar las claves en `.env`

1. Crea una cuenta gratuita en [pinecone.io](https://www.pinecone.io/) y copia tu API key.
2. Obtén tu token de HuggingFace en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Crea el archivo `.env` con ambas claves:

```
PINECONE_API_KEY=pcsk_XXXXXXXXXXXXXXX
HF_TOKEN=hf_XXXXXXXXXXXXXXX
```

> El `HF_TOKEN` se usa para autenticar las llamadas al **HF Router** (extractor de filtros) y evitar el límite de peticiones anónimas.

### 4. Ingestar el PDF en Pinecone + FAISS

```bash
.venv/bin/python3 main.py ingest
```

Esto hará lo siguiente:
1. **Lee** el PDF `Saberes_2ESO.pdf` con PyMuPDF.
2. **Divide** el texto en chunks semánticos por **asignatura** y **trimestre**.
3. **Almacena** en cada chunk:
   - `text` → únicamente el contenido del criterio (texto limpio, sin prefijos).
   - `section` → asignatura (solo como metadato).
   - `trimester` → trimestre (solo como metadato).
   - `page` → número de página de origen.
4. **Genera embeddings** en local con `paraphrase-multilingual-MiniLM-L12-v2`.
5. **Crea el índice** `saberes-2eso` en Pinecone (si no existe) y sube todos los vectores.
6. **Guarda** un índice FAISS local para búsquedas offline.

Salida esperada:
```
📄 Leyendo PDF: /ruta/Saberes_2ESO.pdf
   → 3 páginas con texto.
✂️  Chunking semántico (max=800 chars)...
   → 29 fragmentos | 10 secciones: Educación Física, Física y Química, ...
🤗 Generando embeddings (paraphrase-multilingual-MiniLM-L12-v2)...
🌲 Conectando con Pinecone...
   ℹ️  El índice 'saberes-2eso' ya existe.
🗑️  Limpiando índice anterior...
🚀 Subiendo vectores a Pinecone...
   → 29/29
💾 Índice FAISS guardado en faiss_index/ (29 vectores)
✅ Ingesta completada: 29 fragmentos en 'saberes-2eso'.
```

### 5. Hacer consultas (CLI)

```bash
.venv/bin/python3 main.py query "¿Qué saberes básicos hay en matemáticas en el primer trimestre?"
```

Flujo de la consulta:
1. **Extracción** — `Qwen2.5-0.5B-Instruct` (vía HF Router API) analiza la query con el prompt estructurado e infiere `asignatura` y `trimestre`.
2. **Búsqueda** — La query completa original se vectoriza y se busca en FAISS/Pinecone. Los valores extraídos se aplican como filtros sobre los metadatos.

Salida esperada:
```
🔎 Pregunta: ¿Qué saberes básicos hay en matemáticas en el primer trimestre?
🧠 Extrayendo filtros (HF Router · Qwen/Qwen2.5-0.5B-Instruct)...
   🏷️  Extraídos   → asignatura: matemáticas | trimestre: 1.º trimestre
   🔧 Normalizados → asignatura: matematicas | trimestre: 1 trimestre
🔍 Búsqueda vectorial + filtros estructurados...
⚡ Motor: FAISS (local)
📊 Top 5 resultados:

  [1] (similitud: 0.7523) — Matemáticas | 1.º trimestre | Pág. 1
      Números, operaciones, proporciones y geometría básica...
```

Se puede forzar el motor de búsqueda con `--engine`:

```bash
.venv/bin/python3 main.py query "pregunta" --engine faiss     # Solo FAISS (local)
.venv/bin/python3 main.py query "pregunta" --engine pinecone  # Solo Pinecone (cloud)
.venv/bin/python3 main.py query "pregunta" --engine auto      # FAISS → Pinecone fallback (por defecto)
```

### 6. Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Implementa el **mismo flujo de 2 pasos** que `main.py` (extracción con HF Router + búsqueda FAISS) con:
- Campo de búsqueda semántica con ejemplos en el sidebar
- Detección visual de asignatura y trimestre extraídos
- Filtro de similitud mínima
- Resultados expandibles con sección, trimestre, página y barra de similitud con colores

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
                                         ┌─────────────────────────────┐
                                         │  Chunking semántico          │
                                         │  1. Detecta asignaturas      │
                                         │  2. Detecta trimestres       │
                                         │  3. Agrupa párrafos          │
                                         │  4. Divide por oraciones     │
                                         │  → text: solo el criterio    │
                                         │  → section/trimester: solo   │
                                         │     como metadatos           │
                                         └──────────┬──────────────────┘
                                                    │
                                                    ▼
                                         ┌───────────────────────┐
                                         │ sentence-transformers  │
                                         │ genera embeddings      │
                                         │ (384 dims, en local)   │
                                         └──────┬────────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                             ┌─────────────┐        ┌──────────────────────────┐
                             │  INGESTA:   │        │  CONSULTA (2 pasos):     │
                             │  Pinecone + │        │  1. HF Router API →      │
                             │  FAISS local│        │     extrae asignatura +  │
                             └─────────────┘        │     trimestre de query   │
                                                    │  2. FAISS local →        │
                                                    │     Pinecone (fallback)  │
                                                    │     + filtros metadatos  │
                                                    └──────────────────────────┘
```

1. **Extracción de texto**: PyMuPDF lee el PDF y extrae el texto de cada página.
2. **Chunking semántico (4 niveles)**:
   - **Nivel 1 — Secciones**: detecta nombres de asignaturas como encabezados.
   - **Nivel 2 — Trimestres**: detecta marcas de trimestre y crea un chunk por cada combinación asignatura/trimestre.
   - **Nivel 3 — Párrafos**: agrupa párrafos hasta el límite de 800 caracteres.
   - **Nivel 4 — Oraciones**: divide párrafos largos por oraciones completas (nunca a mitad de frase).
   - **Campo `text`**: contiene únicamente el criterio (texto limpio, sin prefijos ni encabezados).
   - **`section` y `trimester`**: almacenados exclusivamente como metadatos para filtrado posterior.
3. **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, 50+ idiomas) convierte cada chunk en un vector de 384 dimensiones ejecutándose **en local**.
4. **Almacenamiento dual**:
   - **Pinecone**: índice serverless (AWS, us-east-1, métrica coseno).
   - **FAISS**: índice local para búsquedas rápidas y offline.
5. **Consulta — Paso 1 (Extracción)**:
   - `Qwen/Qwen2.5-0.5B-Instruct` se llama vía **HF Router featherless-ai** (API OpenAI-compatible, sin ejecución local).
   - Usa el prompt estructurado para inferir `asignatura` y `trimestre` directamente del texto de la query, sin listas predefinidas ni reglas hardcodeadas.
   - Devuelve un JSON `{"asignatura": ..., "trimestre": ...}` con `null` si un campo no aparece en la query.
6. **Consulta — Paso 2 (Búsqueda)**:
   - La **query original completa** se vectoriza y se busca en FAISS (con fallback a Pinecone).
   - Los valores extraídos se aplican como **filtros sobre los metadatos** (`section` y `trimester`) con normalización robusta (tildes, ordinales, mayúsculas).

---

## 🧠 Prompt del extractor de filtros

```
Actúa como un extractor de datos especializado en currículo educativo.
Tu tarea es analizar la consulta (query) del usuario y extraer dos campos
específicos: trimestre y asignatura.
Reglas de extracción:
- trimestre: Debe seguir el formato "X.º trimestre" (ej. 1.º trimestre, 2.º trimestre).
- asignatura: El nombre de la materia (ej. inglés, matemáticas, historia).
Valores ausentes: Si alguno de estos datos no se menciona explícitamente en la
consulta, asigna el valor null.
Formato de salida: Responde únicamente con un objeto JSON.
Ejemplos:
Entrada: "¿Qué entra en el proyecto oral de inglés del primer trimestre?"
Salida: {"trimestre": "1.º trimestre", "asignatura": "inglés"}
Entrada: "Dime los criterios de evaluación de matemáticas."
Salida: {"trimestre": null, "asignatura": "matemáticas"}
```

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

- La primera ejecución descargará el modelo `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) en caché local. Las siguientes usan la caché.
- El extractor de filtros (`Qwen2.5-0.5B-Instruct`) se ejecuta **completamente en la nube** vía HF Router; no requiere GPU ni descarga de pesos localmente.
- Pinecone tier gratuito permite 1 índice serverless con hasta 2 GB de almacenamiento.
- Si quieres reingestar el PDF, ejecuta de nuevo `ingest`; los vectores se sobreescriben.
- Los filtros de metadatos son opcionales: si la query no menciona asignatura ni trimestre, se realiza una búsqueda semántica global sin filtros.
