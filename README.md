# FinDoc Analyser — AI-Powered Financial Document Analysis

A RAG (Retrieval-Augmented Generation) web application that allows users to upload financial PDFs and query them using natural language. Built with Flask, LangChain, ChromaDB, and OpenAI — containerised with Docker, stored in Azure Container Registry, and deployed on Azure Web App with CI/CD via GitHub Actions.

---

## Features

- **PDF Ingestion** — Upload financial documents (PDFs) which are chunked and stored as vector embeddings in ChromaDB
- **AI-Powered Q&A** — Ask natural language questions about the document; the LLM follows a structured 6-step chain-of-thought (START → PLAN → SEARCH → READ → ANALYSE → OUTPUT)
- **Finance-Domain Restriction** — The system prompt enforces that only finance-related questions are answered
- **Session-Based Auth** — Lightweight secret-key login to protect access
- **Case ID Isolation** — Each uploaded document is stored under a unique Case ID, keeping documents separate
- **Document Deletion** — Users can delete a document and its vector store collection at any time
- **Async Pipeline** — Heavy PDF parsing and embedding operations run in background threads to keep the web server responsive

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| AI / LLM | OpenAI (GPT) via LangChain |
| Embeddings | OpenAI Embeddings |
| Vector Store | ChromaDB (persistent) |
| PDF Parsing | PyMuPDF (fitz) |
| Containerisation | Docker |
| Container Registry | Azure Container Registry (ACR) |
| Hosting | Azure Web App |
| CI/CD | GitHub Actions |

---

## Project Structure

```
.
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipeline
├── src/
│   └── pipeline/
│       └── rag_pipeline.py # Core RAG logic: ingest, query, delete
├── templates/              # Jinja2 HTML templates (login, index, home)
├── .gitignore
├── Dockerfile
├── README.md
├── app.py                  # Flask app — routes and API endpoints
├── config.py               # Settings (OpenAI key, Chroma path, model names, etc.)
└── requirements.txt
```

---

## Getting Started (Local)

### Prerequisites

- Python 3.10+
- An OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/ShibjiRout/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file or set the following environment variables (see `config.py` for all settings):

```env
OPENAI_API_KEY=sk-...
API_SECRET_KEY=your-secret-login-key
CHROMA_PATH=./chroma_db
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

### 4. Run the app

```bash
python app.py
```

The app will be available at `http://localhost:8000`.

---

## Running with Docker

### Build the image

```bash
docker build -t findoc-analyser .
```

### Run the container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e API_SECRET_KEY=your-secret \
  findoc-analyser
```

---

## Azure Deployment

This project is deployed to **Azure Web App** using a Docker image hosted in **Azure Container Registry (ACR)**.

### Infrastructure Overview

```
GitHub Repository
      │
      │  push to main
      ▼
GitHub Actions (CI/CD)
      │
      ├─── docker build
      ├─── docker push ──► Azure Container Registry (ACR)
      │
      └─── Azure Web App pulls latest image from ACR
```

### CI/CD Pipeline

The `.github/workflows/` directory contains the GitHub Actions workflow that:

1. Builds the Docker image on every push to `main`
2. Pushes the image to Azure Container Registry
3. Triggers the Azure Web App to pull and deploy the new image

### Required GitHub Secrets

Set these in your GitHub repository under **Settings → Secrets and variables → Actions**:

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | Azure service principal credentials (JSON) |
| `REGISTRY_LOGIN_SERVER` | ACR login server (e.g. `yourregistry.azurecr.io`) |
| `REGISTRY_USERNAME` | ACR username |
| `REGISTRY_PASSWORD` | ACR password |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `API_SECRET_KEY` | The login secret key for the app |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET/POST` | `/login` | Login page |
| `GET` | `/logout` | Clear session and redirect to login |
| `GET` | `/` | Upload page (index) |
| `GET` | `/home?case_id=<id>` | Document Q&A page |
| `POST` | `/upload` | Upload a PDF and ingest it into ChromaDB |
| `POST` | `/ask` | Ask a question about a document |
| `POST` | `/delete` | Delete a document's vector store collection |

---

## How It Works

1. **Upload** — The user enters a Case ID and uploads a PDF. The app parses the PDF with PyMuPDF, splits the text into chunks, embeds them with OpenAI, and stores them in a ChromaDB collection named after the Case ID.

2. **Query** — The user asks a question. The app retrieves the top-K most relevant chunks from ChromaDB, passes them as context to the LLM, and the LLM responds using a structured 6-step reasoning process.

3. **Delete** — The user can delete the case, which removes the ChromaDB collection to respect privacy.

---

## Notes

- Maximum upload file size is **16 MB**
- Only `.pdf` files are accepted
- The LLM is restricted to finance-related questions only. Non-financial queries return: `"Sorry, please ask a finance-related question."`
- Each Case ID maps to a separate ChromaDB collection, so multiple documents can coexist without interference