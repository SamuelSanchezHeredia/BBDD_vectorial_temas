# 📚 Base de Datos Vectorial — Saberes Básicos 2º ESO

Base de datos vectorial con **Pinecone** y **HuggingFace** que almacena el contenido del archivo `Saberes_2ESO.pdf` y permite hacer consultas por similitud semántica.

---

## 🧰 Tecnologías

| Tecnología | Uso |
|---|---|
| [Pinecone](https://www.pinecone.io/) | Base de datos vectorial en la nube (serverless) |
| [sentence-transformers](https://www.sbert.net/) | Modelo de embeddings de HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, multilingüe) |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracción de texto del PDF |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Gestión de variables de entorno |

---

## 📁 Estructura del proyecto

```
BBDD_vectorial_pinecone/
├── .env                 # Clave API de Pinecone (NO subir a Git)
├── .gitignore           # Excluye .env, .venv/, __pycache__/
├── .venv/               # Entorno virtual de Python
├── main.py              # Script principal (ingesta + consulta)
├── requirements.txt     # Dependencias del proyecto
├── Saberes_2ESO.pdf     # PDF con los saberes básicos de 2º ESO
└── README.md            # Este archivo
```

---

## 🚀 Puesta en marcha

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd BBDD_vectorial_pinecone
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

### 4. Ingestar el PDF en Pinecone

```bash
.venv/bin/python3 main.py ingest
```

Esto hará lo siguiente:
1. **Lee** el PDF `Saberes_2ESO.pdf` con PyMuPDF.
2. **Divide** el texto en chunks semánticos: detecta los encabezados de asignatura y agrupa el contenido de cada una (los 3 trimestres) en un único fragmento con sentido completo.
3. **Genera embeddings** en local con el modelo `all-MiniLM-L6-v2` de HuggingFace.
4. **Crea el índice** `saberes-2eso` en Pinecone (si no existe).
5. **Sube** todos los vectores con sus metadatos (texto, número de página y nombre de sección).

Salida esperada:
```
📄 Leyendo PDF: /ruta/Saberes_2ESO.pdf
   → 3 páginas con texto.
✂️  Aplicando chunking semántico (max=800 chars por chunk)...
   → 10 fragmentos generados.
   → 10 secciones detectadas: Educación Física, Física y Química, ...
🤗 Cargando modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2
🔢 Generando embeddings para 10 fragmentos...
🌲 Conectando con Pinecone...
   ℹ️  El índice 'saberes-2eso' ya existe.
🗑️  Limpiando índice anterior...
🚀 Subiendo vectores a Pinecone...
   → Subidos 10/10
✅ Ingesta completada: 10 fragmentos en el índice 'saberes-2eso'.
```

### 5. Hacer consultas

```bash
.venv/bin/python3 main.py query "¿Qué saberes básicos hay en matemáticas?"
```

Devuelve los 5 fragmentos más relevantes ordenados por similitud:

```
🔎 Pregunta: ¿Qué saberes básicos hay en matemáticas?
📊 Top 5 resultados:

  [1] (similitud: 0.7523) — Página 2
      Matemáticas: Resolución de problemas...

  [2] (similitud: 0.6891) — Página 2
      Geometría y medida. Cálculo de áreas...
```

---

## ⚙️ Cómo funciona

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Saberes_2ESO │────▶│  PyMuPDF extrae   │────▶│  Texto plano │
│    .pdf      │     │  texto por página  │     │  por página  │
└──────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
                                         ┌────────────────────────┐
                                         │  Chunking semántico     │
                                         │  1. Detecta asignaturas │
                                         │  2. Agrupa párrafos     │
                                         │  3. Divide por oraciones│
                                         │  → 1 chunk/asignatura   │
                                         └──────────┬─────────────┘
                                                    │
                                                    ▼
                                         ┌───────────────────┐
                                         │ sentence-transformers │
                                         │ genera embeddings    │
                                         │ (384 dimensiones)    │
                                         └──────┬────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                             ┌─────────────┐        ┌─────────────┐
                             │  INGESTA:   │        │  CONSULTA:  │
                             │  upsert en  │        │  query por  │
                             │  Pinecone   │        │  similitud  │
                             └─────────────┘        └─────────────┘
```

1. **Extracción**: PyMuPDF lee el PDF y extrae el texto de cada página.
2. **Chunking semántico (3 niveles)**:
   - **Nivel 1 — Secciones**: detecta los nombres de asignaturas como encabezados y los usa para separar el contenido.
   - **Nivel 2 — Párrafos**: dentro de cada sección, agrupa párrafos juntos hasta el límite de 800 caracteres. Así cada chunk contiene el contexto completo de una asignatura (los 3 trimestres).
   - **Nivel 3 — Oraciones**: si un párrafo supera el límite, se divide por oraciones completas (nunca a mitad de frase).
3. **Embeddings**: el modelo `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) convierte cada chunk en un vector de 384 dimensiones. Es multilingüe (50+ idiomas, incluido español) y se ejecuta **en local**, sin llamadas a APIs externas.
4. **Almacenamiento**: los vectores se suben a un índice serverless de Pinecone (AWS, us-east-1, métrica coseno) con metadatos de texto, página y nombre de sección.
5. **Consulta**: la pregunta del usuario se convierte en un embedding con el mismo modelo y se buscan los vectores más cercanos en Pinecone (top 5 por defecto), mostrando la sección y página de cada resultado.

---

## 📋 Comandos disponibles

| Comando | Descripción |
|---|---|
| `python main.py ingest` | Sube el contenido del PDF a Pinecone |
| `python main.py query "pregunta"` | Busca fragmentos relevantes por similitud |

---

## 📌 Notas

- La primera ejecución descargará el modelo `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB). Las siguientes usan la caché local.
- Pinecone tier gratuito permite 1 índice serverless con hasta 2 GB de almacenamiento.
- Si quieres reingestar el PDF, puedes volver a ejecutar `ingest`; los vectores se sobreescriben (mismos IDs).

