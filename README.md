# Local RAG Chatbot

A practical, local-first Retrieval-Augmented Generation (RAG) application for
asking questions about your own documents.

The current application is a standalone Streamlit app. It runs the retrieval
pipeline in the same Python process as the user interface:

```text
Documents
   │
   ▼
Loaders ──► Chunks ──► Local Hugging Face embeddings ──► Chroma
                                                        │
Question ──► Question embedding ──► Similarity retrieval┘
                                                        │
                                                        ▼
                         Context-aware prompt ──► Groq ──► Answer
```

The project uses:

- **Streamlit** for the user interface
- **LangChain document loaders** for PDF, DOCX, CSV, JSON, TXT, and web pages
- **Hugging Face `sentence-transformers`** for local embeddings
- **Chroma** for persistent vector storage
- **Groq** for fast chat completion
- **`openai/gpt-oss-120b`** as the current default Groq model

No FastAPI server or Ollama server is required by the current Streamlit app.
The older `main.py` FastAPI application remains in the repository for
reference, but the recommended path is `streamlit_app.py`.

---

## Features

- Upload PDF, DOCX, CSV, JSON, and TXT documents.
- Index a web page by URL.
- Split documents into searchable chunks.
- Persist embeddings in a local Chroma database.
- Retrieve the most relevant chunks for each question.
- Ask grounded questions through Groq.
- Keep chat history in the Streamlit session.
- Select the number of retrieved chunks.
- Choose a document type so the appropriate prompt template can be used.
- Reuse the RAG pipeline directly from Python without Streamlit.
- Display useful errors for missing dependencies, missing credentials, invalid
  models, unsupported files, and failed API calls.

---

## Project layout

```text
rag-project/
├── streamlit_app.py              # Main standalone Streamlit application
├── main.py                       # Legacy FastAPI application
├── requirements.txt              # Python dependencies
├── .env                          # Local secrets and model configuration
├── chroma/                       # Persistent Chroma data (created at runtime)
├── src/
│   ├── chunking/
│   │   └── chunker.py            # Text splitting
│   ├── embeddings/
│   │   └── embedder.py           # Local Hugging Face embeddings
│   ├── ingestion/
│   │   └── loader.py             # PDF/DOCX/CSV/JSON/TXT/web loaders
│   ├── llm/
│   │   └── llm_client.py         # Groq client and answer generation
│   ├── prompts/
│   │   └── prompt_templates.py   # Document-aware prompt templates
│   ├── retrieval/
│   │   └── retriever.py          # Similarity retrieval
│   ├── utils/
│   │   └── helpers.py            # RAGPipeline orchestration
│   └── vectordb/
│       └── vector_store.py       # Chroma wrapper
└── tests/                        # Project tests
```

---

## Requirements

- Linux, macOS, or Windows
- Python 3.10 or newer
- A Groq account and API key
- Internet access for:
  - the first Hugging Face embedding-model download
  - Groq API requests

The embedding model runs locally after it has been downloaded. Groq is used
only for answer generation.

---

## Installation

### 1. Open the project directory

```bash
cd /home/abdullah-saeed/rag-project
```

Use your own project path if it is different.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

If `.venv` already exists, do not recreate it.

### 3. Activate the environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

Using `uv`:

```bash
uv pip install -r requirements.txt
```

Using regular `pip`:

```bash
python -m pip install -r requirements.txt
```

If your shell cannot find `python` after activation, use the absolute project
interpreter:

```bash
/home/abdullah-saeed/rag-project/.venv/bin/python -m pip install -r requirements.txt
```

---

## Environment configuration

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Do not commit `.env`, paste the key into source code, or expose it in
screenshots. If a key has been exposed, revoke it in the Groq dashboard and
create a replacement.

### Choosing a Groq model

The model must be available to your Groq account. The app currently defaults
to:

```text
openai/gpt-oss-120b
```

You can override it in `.env`:

```env
GROQ_MODEL=openai/gpt-oss-20b
```

You can also change the model in the Streamlit sidebar. If Groq returns
`model_not_found`, list the models available to your account and select one of
those model IDs:

```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
print([model.id for model in client.models.list().data])
```

---

## Run the Streamlit application

From the project root:

```bash
.venv/bin/streamlit run streamlit_app.py
```

Or use the absolute path:

```bash
/home/abdullah-saeed/rag-project/.venv/bin/streamlit run \
  /home/abdullah-saeed/rag-project/streamlit_app.py
```

Open the URL shown by Streamlit, normally:

```text
http://localhost:8501
```

If port `8501` is already in use:

```bash
.venv/bin/streamlit run streamlit_app.py --server.port 8502
```

### Recommended first-use flow

1. Start Streamlit.
2. Check the sidebar model. Use a model available to your Groq account.
3. Leave the collection name as `documents`, unless you need separate
   collections.
4. Leave the Chroma path as `./chroma` to use the default persistent database.
5. Upload a document.
6. Click **Index uploaded file**.
7. Wait for the indexing confirmation.
8. Ask a question in the chat box.

Indexing must happen before asking questions. If no documents have been
indexed, retrieval cannot provide context for the answer.

---

## How the RAG pipeline works

### 1. Ingestion

`src/ingestion/loader.py` converts source files into LangChain `Document`
objects. A document contains:

```python
Document(
    page_content="text extracted from the source",
    metadata={"source": "example.pdf"},
)
```

Supported sources:

| Source | Loader |
|---|---|
| PDF | `PyPDFLoader` |
| DOCX | `Docx2txtLoader` |
| CSV | `CSVLoader` |
| JSON | Project JSON loader |
| TXT | `TextLoader` |
| Web page | `AsyncHtmlLoader` |

The main helper is:

```python
from src.ingestion import fallback

documents = fallback(file_path="notes.pdf")
web_documents = fallback(url="https://example.com")
```

Only one of `file_path` and `url` should be supplied at a time.

### 2. Chunking

Large documents are split into smaller pieces before embedding. The default
pipeline uses:

```text
chunk_size = 1000
chunk_overlap = 200
```

Overlap helps preserve meaning when an answer spans a chunk boundary.

### 3. Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model runs locally through `HuggingFaceEmbeddings`. The first run may take
time because model weights must be downloaded. Later runs reuse the local
cache.

### 4. Vector storage

Chroma stores each chunk, its embedding, metadata, and source text. The
default database directory is:

```text
./chroma
```

The database persists between application restarts. Do not delete the
`chroma/` directory unless you intentionally want to rebuild the index.

### 5. Retrieval

When a question is submitted:

1. The question is embedded using the same embedding model.
2. Chroma searches for the nearest stored chunks.
3. The top `k` chunks are selected.
4. The chunks are formatted as context.

The Streamlit sidebar allows `k` from 1 to 10. A higher value provides more
context but can add noise and increase the prompt size.

### 6. Prompting and generation

The retrieved chunks are passed to a document-aware prompt template. Groq
then generates the answer using the configured model.

The LLM client is implemented in `src/llm/llm_client.py` and uses:

```python
from src.llm import get_llm, generate_answer
```

No Ollama configuration is needed for this project.

---

## Use the RAG components from Python

The main reusable entry point is `RAGPipeline` in `src/utils/helpers.py`.
This lets another Python script use ingestion, chunking, embeddings, Chroma,
retrieval, prompts, and Groq without importing Streamlit.

### Build a pipeline

```python
from src.utils import build_rag_pipeline

pipeline = build_rag_pipeline(
    collection_name="documents",
    db_type="persistent",
    db_path="./chroma",
)
```

The collection is created if it does not already exist.

### Index a local document

```python
from src.ingestion import fallback

documents = fallback(file_path="data/student_record.pdf")
ids = pipeline.index_documents(
    documents,
    source_type="pdf",
)

print(f"Indexed {len(ids)} chunks")
```

For a DOCX file:

```python
documents = fallback(file_path="data/profile.docx")
pipeline.index_documents(documents, source_type="docx")
```

For a web page:

```python
documents = fallback(url="https://example.com")
pipeline.index_documents(documents, source_type="web")
```

### Retrieve relevant documents

```python
matches = pipeline.retrieve(
    "What is the student's SGPA in the second semester?",
    k=4,
)

for document in matches:
    print(document.page_content)
    print(document.metadata)
```

### Ask a grounded question

```python
answer = pipeline.ask(
    "What is the student's SGPA in the second semester?",
    document_type="pdf",
    k=4,
    model="openai/gpt-oss-120b",
)

print(answer)
```

The answer call performs retrieval first and then sends the retrieved context
to Groq. It does not send the entire original document to Groq.

### Use the lower-level Groq client

```python
from src.llm import generate_answer, get_llm

llm = get_llm(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1500,
)

answer = generate_answer(
    question="What language does he know best?",
    context="The student speaks English and Urdu fluently.",
    document_type="generic",
    llm=llm,
)

print(answer)
```

### Customize chunking

```python
ids = pipeline.index_documents(
    documents,
    source_type="pdf",
    chunk_size=700,
    chunk_overlap=100,
)
```

Smaller chunks can improve precision for structured records. Larger chunks can
preserve more surrounding explanation. Adjust these values based on the
documents rather than changing them unnecessarily.

---

## Use the vector store directly

For lower-level Chroma operations:

```python
from src.vectordb import VectorDBOperations

store = VectorDBOperations(db_type="persistent")
store.connect(path="./chroma")
collection = store.create_collection("documents")
```

The wrapper supports:

```python
store.insert_vector(
    vector_id="example-1",
    vector_data=[0.1, 0.2, 0.3],
    metadata={"source_type": "txt"},
    document="Example text",
)

store.get_vectors(limit=10)
store.query_vectors(query_embedding=[0.1, 0.2, 0.3], n_results=4)
store.update_vector("example-1", document="Updated text")
store.delete_vector("example-1")
```

In normal application code, prefer `RAGPipeline` because it handles chunking,
embedding, IDs, metadata, and retrieval consistently.

---

## Prompt templates

Prompt templates live in `src/prompts/prompt_templates.py`. They support
document types such as:

```text
generic, pdf, docx, csv, json, txt, web, database
```

Use them directly:

```python
from src.prompts import format_prompt, get_prompt_template

template = get_prompt_template("pdf")
prompt = format_prompt(
    question="What is the CGPA?",
    context="CGPA: 8.42",
    document_type="pdf",
)
```

The templates instruct the model to answer from the supplied context instead
of inventing unsupported facts. A good RAG answer depends on both retrieval
quality and prompt quality.

---

## Common problems and fixes

### `GROQ_API_KEY is missing`

Confirm that `.env` is in the project root:

```text
rag-project/.env
```

It should contain:

```env
GROQ_API_KEY=your_key_here
```

Then stop and restart Streamlit. Environment variables are loaded when the
application starts.

### `model_not_found` or HTTP 404 from Groq

The configured model is unavailable to the current Groq account. List the
available models and update `GROQ_MODEL`:

```python
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
print([item.id for item in client.models.list().data])
```

Use an actual chat model ID from that list.

### `No module named docx2txt`

Install DOCX support:

```bash
uv pip install docx2txt
```

### `pypdf package not found`

Install PDF support:

```bash
uv pip install pypdf
```

### Hugging Face embedding errors

Install the local embedding dependencies:

```bash
uv pip install langchain-huggingface sentence-transformers
```

The first successful embedding call can download model files. Allow it time
to complete and make sure the machine has internet access.

### Port already in use

Use another port:

```bash
.venv/bin/streamlit run streamlit_app.py --server.port 8502
```

Or stop the old Streamlit process before starting a new one.

### The answer is slow

The first indexing operation can be slow because the embedding model is being
loaded or downloaded. Later operations should be faster. To reduce response
time:

- Use a smaller `k`, such as 3 or 4.
- Use shorter chunks when documents are highly structured.
- Use a faster Groq model available to your account.
- Avoid indexing the same large document repeatedly.

### The answer says the information is not in the context

Check that:

1. The document was indexed successfully.
2. The question uses terms present in the document.
3. The selected document type is appropriate.
4. `k` is large enough.
5. The Chroma path and collection name are the same between indexing and
   questioning.

---

## Security notes

- Keep `.env` private.
- Never hardcode `GROQ_API_KEY`.
- Rotate a key immediately if it is exposed.
- Do not commit the `chroma/` directory if it contains sensitive documents.
- Treat uploaded documents as private data.
- Use a separate Chroma path or collection for different datasets.
- Review web URLs before indexing them.

---

## FastAPI note

`main.py` contains an earlier FastAPI integration with endpoints such as
health, ingestion, upload ingestion, and question answering. The current
recommended application does not need it:

```bash
.venv/bin/streamlit run streamlit_app.py
```

Use the FastAPI application only if you specifically need an HTTP API and are
prepared to maintain its dependency and startup path separately.

---

## Development checks

Check Python syntax:

```bash
.venv/bin/python -m compileall streamlit_app.py src
```

Check the installed Groq configuration without making a completion request:

```bash
.venv/bin/python -c \
  "from src.llm import get_llm; print(get_llm().model)"
```

Check that the local embedding provider can be constructed:

```bash
.venv/bin/python -c \
  "from src.embeddings import get_embeddings; print(type(get_embeddings()).__name__)"
```

Run tests, when present:

```bash
.venv/bin/python -m pytest
```

---

## Typical end-to-end example

```python
from src.ingestion import fallback
from src.utils import build_rag_pipeline

pipeline = build_rag_pipeline(
    collection_name="student-documents",
    db_path="./chroma",
)

documents = fallback(file_path="student_record.pdf")
pipeline.index_documents(documents, source_type="pdf")

answer = pipeline.ask(
    "What is the student's CGPA?",
    document_type="pdf",
    k=4,
    model="openai/gpt-oss-120b",
)

print(answer)
```

This is the same core flow used by the Streamlit application:

```text
load → chunk → embed → store → retrieve → prompt → Groq answer
```

---

## License and responsibility

This project is intended as an educational and practical local RAG example.
Check the licenses and terms of the models, libraries, source documents, and
Groq services used with your deployment.
