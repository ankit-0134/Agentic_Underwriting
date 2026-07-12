# Underwriting

An AI-assisted insurance underwriting tool. Upload application, financial, and
medical documents for a case and get back structured data extraction plus a
RAG-grounded risk dashboard (LOW / MEDIUM / HIGH classifications with reasons,
and cross-document mismatch checks for name, age, and income).

## How it works

1. **Extraction** — Uploaded files (PDF, JPG, PNG) are routed through one of
   two paths:
   - Digital PDFs (text layer present) → parsed directly with `pdfplumber`.
   - Scanned PDFs / images → rendered to an image and sent to `gpt-4o` vision
     to OCR the text.
2. **Structured extraction** — The raw text from each document category
   (application, financial, medical) is sent to `gpt-4o` with a category-
   specific prompt that returns a fixed JSON schema (personal details,
   occupation, avocations, lifestyle, health, financials, lab results, etc.),
   including derived fields like income coverage ratio, EMI-to-income ratio,
   and income consistency flags.
3. **Risk classification (RAG)** — The extracted fields are matched against a
   Chroma vector store of underwriting rules (built by `rag_setup.py`) to
   classify each risk feature as LOW/MEDIUM/HIGH with a rule-grounded reason,
   and to compute mismatch checks across documents.
4. **Dashboard** — The frontend renders the combined payload as an
   underwriting summary dashboard.

## Project structure

```
Underwriting/
├── backend/
│   ├── main.py            # FastAPI app: /extract and /analyse endpoints
│   ├── rag_setup.py        # Builds/seeds the Chroma collections (underwriting rules)
│   ├── rag_classify.py     # Queries Chroma + classifies risk, computes mismatches
│   ├── test_rag_query.py   # Manual script for testing RAG queries
│   ├── requirements.txt
│   ├── chroma_db/          # Persisted vector store (generated, gitignored)
│   └── .env                # OPENAI_API_KEY, CHROMA_DIR (not committed)
└── frontend/
    ├── src/
    │   ├── App.jsx          # Upload UI, calls backend /extract and /analyse
    │   ├── Dashboard.jsx     # Risk dashboard rendering
    │   └── data/mockCase.json
    ├── package.json
    └── vite.config.js
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI API key with access to `gpt-4o` and `text-embedding-3-small`
- [Poppler](https://poppler.freedesktop.org/) for PDF→image conversion
  (scanned PDF path uses `pdf2image`, which shells out to Poppler's
  `pdftoppm`). On Windows, install Poppler and update the `poppler_path` in
  `backend/main.py` (`extract_with_vision`) to match your install location.

## Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-...
CHROMA_DIR=C:\path\to\Underwriting\backend\chroma_db
```

Build the vector store of underwriting rules (only needed once, or whenever
`rag_setup.py`'s rule content changes):

```bash
python rag_setup.py
```

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

The backend expects requests from `http://localhost:5173` (see CORS config
in `main.py`) — update `allow_origins` if the frontend runs elsewhere.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:5173` by default (Vite). The backend base URL is
currently hardcoded in `src/App.jsx` as `http://localhost:8000`.

## API

### `POST /extract`
Extracts raw text from a single uploaded file.

- **Body**: `multipart/form-data`, field `file`
- **Response**: `{ filename, extraction_method, extracted_text }`

### `POST /analyse`
Runs the full pipeline for a case: extraction + structured field parsing +
RAG risk classification, across up to three document categories.

- **Body**: `multipart/form-data` with optional repeated fields:
  - `application_files`
  - `financial_files`
  - `medical_files`
- **Response**: `{ application, financial, medical, ...dashboard_payload }`

### `GET /`
Health check — returns `{ "status": "Underwriting Backend Running" }`.

## Environment variables

| Variable         | Where          | Purpose                                   |
|------------------|----------------|--------------------------------------------|
| `OPENAI_API_KEY` | `backend/.env` | Auth for OpenAI extraction/classification calls |
| `CHROMA_DIR`     | `backend/.env` | Path to the persisted Chroma vector store |

`.env` is gitignored — never commit real keys. Rotate the key immediately if
it's ever exposed in a commit, log, or screenshot.

## Notes / known limitations

- The Poppler path in `main.py` is hardcoded to a local Windows install path
  — adjust for your machine or make it configurable via env var if this
  needs to run cross-platform or in CI/containers.
- The frontend backend URL (`API_URL` in `App.jsx`) is hardcoded rather than
  read from a Vite env var — fine for local dev, but should move to
  `import.meta.env.VITE_API_URL` before deploying anywhere beyond localhost.
- `chroma_db/` is a local persisted store; it isn't committed, so a fresh
  clone must run `python rag_setup.py` before `/analyse` will work.
