# Rescue AI Post-Adoption FAQ API

FastAPI backend for the Dogs on a Journey post-adoption FAQ chatbot. The API searches a Dutch adoption-guide corpus and returns a grounded answer with section and page citations.

## Live demo

- FAQ website: https://liajohn18.github.io/RescueAI_match_test/faq.html
- API base URL: https://rescueai-post-adoption-api.onrender.com
- Interactive API docs: https://rescueai-post-adoption-api.onrender.com/docs
- Ask endpoint: `POST /api/ask`

Example request:

```json
{"question":"Hoe pak ik wandelen en zindelijkheid aan?"}
```

The response includes an `answer` and `sources` array containing source titles, page numbers, and retrieval scores. The API root `/` does not serve a homepage; use `/docs` to inspect the API.

## How it works

1. `post_adoption_rag/ingest.py` extracts adoption-guide sections and page metadata into `data/post_adoption_sections.json`.
2. `post_adoption_rag/hybrid_search.py` combines BM25 keyword search with multilingual semantic search using reciprocal rank fusion.
3. `api.py` uses OpenAI to select the best supporting section and generate an answer grounded in that section.
4. The static FAQ page sends questions to `/api/ask` and displays the answer and source links.

The FAQ website is a separate static GitHub Pages repository (`RescueAI_match_test`). This repository contains the Python API only.

## Run locally on Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the embedding model into the local cache used by the API:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', cache_folder='.model_cache')"
```

Set your OpenAI API key in the current PowerShell session. Keep the real key private; never commit it or paste it into the website code:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Start the API:

```powershell
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` to inspect the local API. The local FAQ website must use the same local API URL while developing; the published site uses the hosted HTTPS API URL.

## Deployment on Render

The API is deployed as a Render Web Service from the `main` branch. Configure `OPENAI_API_KEY` in the Render service's Environment settings, not in source code.

- Build command:

  ```sh
  pip install -r requirements.txt && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', cache_folder='.model_cache')"
  ```

- Start command:

  ```sh
  uvicorn api:app --host 0.0.0.0 --port $PORT
  ```

Render automatically deploys changes pushed to `main`. Have contributors work on a branch and submit a pull request so changes can be reviewed before merging.

## Evaluation

`post_adoption_rag/evaluation_questions.json` contains evaluation questions, expected sources, and answer-review checklists. Run `python -m post_adoption_rag.evaluate_api` while the API is running locally to test whether the expected source sections and pages are returned. You can run selected cases with `--ids q01 q20`.

This script evaluates source matching; it does **not** automatically judge answer accuracy, language quality, citation correctness, or safety. Review answers against each case's checklist manually. The last observed 20/20 source-match result is not a guarantee that every future answer is correct.

## Current limitations and safety

- The guide-derived content and evaluation checklists require ongoing human review. Retrieval success alone does not prove an answer is correct.
- No human escalation workflow is currently connected to this API. Urgent, injury-related, medical, or unsupported questions should be reviewed by an appropriate person rather than treated as answered by the bot.
- Do not rely on the chatbot for diagnosis, medication, dosage, or emergency advice. Contact a veterinarian or emergency service when appropriate.
- Before broad public use, add request limits and usage monitoring, review/approve the source corpus, and verify answer quality and escalation behavior end to end.

## Repository map

- `api.py` — FastAPI app and `/api/ask` endpoint
- `data/post_adoption_sections.json` — extracted guide sections with page metadata
- `post_adoption_rag/` — ingestion, retrieval, prompts, and evaluation materials
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes local environments, embedding cache, and Python bytecode

