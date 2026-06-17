# FitMind AI Service

FastAPI AI microservice for the FitMind gym management and fitness platform.

This service provides retrieval-augmented generation (RAG) features for exercise search, food/nutrition search, training plan generation, nutrition plan generation, plan modification, and assistant/chat behavior. It syncs exercise and food records from the FitMind MySQL database into a persistent ChromaDB vector store, retrieves relevant records semantically, and uses an LLM to return structured, source-aware responses for the Laravel backend.

The service is safety-aware: it considers user profile data, goals, injuries, food allergies, disliked foods, macro targets, retrieved source records, and fallback guardrails before returning plans or advice. It is part of the FitMind graduation project.

## Main Features

- FastAPI REST service with CORS configuration.
- Liveness endpoints at `GET /` and `GET /health`.
- MySQL-backed sync for exercises and foods.
- Persistent ChromaDB vector store for exercise and nutrition collections.
- Gemini embedding support through `langchain-google-genai`.
- Semantic exercise search with filters for difficulty, muscle group, goal tags, and junk-data exclusion.
- Semantic food search with filters for category, protein, calories, goal tags, and junk-data exclusion.
- RAG-based training plan generation.
- RAG-based nutrition plan generation.
- RAG-based training plan modification.
- RAG-based nutrition plan modification.
- FitMind Assistant chat endpoint with exercise/nutrition retrieval when relevant.
- Progress analysis endpoint using deterministic metrics and heuristics.
- Injury-aware training guidance, including additional shoulder-pain safety guards.
- Nutrition guardrails for allergies, disliked foods, meal count, macro targets, calorie floors, and source validation.
- Structured JSON response parsing and Pydantic validation for LLM outputs.
- Source tracking in search, chat, training, and nutrition responses.
- Gemini generation support with response schemas.
- Optional OpenRouter generation support through the shared LLM client.
- Gemini retry handling for transient generation failures.
- Rule-based fallback behavior for chat, training plans, nutrition plans, and modification flows when RAG/LLM generation fails.

## Tech Stack

- Python 3.11, as used by the Docker image.
- FastAPI.
- Uvicorn.
- Pydantic v2.
- PyMySQL for MySQL access.
- ChromaDB persistent vector database.
- LangChain Google GenAI embeddings.
- Google Gemini API for embeddings and generation.
- OpenRouter chat completions support through `httpx`.
- `python-dotenv` for environment configuration.
- Docker.

This codebase does not use SQLAlchemy or sentence-transformers. Embeddings are generated with Google Generative AI embeddings.

## Project Structure

```text
.
|-- main.py                    # FastAPI app, endpoints, DB sync, Chroma search, RAG flows, fallbacks, guards
|-- config.py                  # Environment loading and typed settings
|-- llm_client.py              # Gemini/OpenRouter chat client abstraction
|-- prompt_builders.py         # Prompt construction for chat, training, and nutrition RAG
|-- rag_schemas.py             # Pydantic models and Gemini-compatible response schemas
|-- response_parsers.py        # JSON extraction and response validation helpers
|-- requirements.txt           # Python dependencies
|-- Dockerfile                 # Container image for the FastAPI service
|-- .env.example               # Environment variable template with placeholder values
|-- chroma_gym/                # Local persistent Chroma data directory
|-- data/sync_checkpoint.json  # Sync checkpoint file
|-- gym_manifest.json          # Exercise vector manifest and fingerprints
`-- nutrition_manifest.json    # Food vector manifest and fingerprints
```

There are no separate router or service packages in the current repository. Most application logic is implemented in `main.py`, with prompts, schemas, response parsing, configuration, and LLM calls split into helper modules.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic service status response. |
| `GET` | `/health` | Liveness check returning `{"status": "healthy"}`. |
| `POST` | `/sync-all?full_sync=false` | Sync exercises and nutrition data from MySQL into ChromaDB. Use `full_sync=true` for a full resync. |
| `POST` | `/search-exercises` | Semantic exercise search using query/body parameters and optional filters. |
| `POST` | `/search-foods` | Semantic food/nutrition search using query/body parameters and optional filters. |
| `POST` | `/generate-training-plan` | Generate a training plan using retrieved exercise context and an LLM, with rule-based fallback. |
| `POST` | `/modify-training-plan` | Modify an existing training plan using retrieved exercise context and safety guards, with fallback. |
| `POST` | `/generate-nutrition-plan` | Generate a nutrition plan using retrieved food context and an LLM, with rule-based fallback. |
| `POST` | `/modify-nutrition-plan` | Modify an existing nutrition plan using retrieved food context, macro/allergy guards, and fallback. |
| `POST` | `/chat` | Assistant endpoint for member-facing fitness and nutrition chat. Uses RAG when the message is exercise or nutrition related. |
| `POST` | `/analyze-progress` | Analyze progress data and suggest whether the current plan should continue or be modified. |

### Shared Request Models

`UserSummary` is used by training, nutrition, modification, and progress endpoints:

- `user_id`
- `level`
- `goal`
- `training_age_years`
- `injuries`
- `weak_points`
- `status`
- `progress_rate`
- `consistency_score`
- `weight`
- `height`
- `age`

Search endpoints accept either query parameters or a JSON body with fields such as:

- `query`
- `n_results`
- `difficulty`
- `muscle_group`
- `category`
- `min_protein`
- `max_calories`
- `goal`
- `exclude_junk`
- `debug_context`

Plan generation and modification endpoints return structured JSON that can include `generation_mode`, `fallback_reason`, `sources`, `injury_warnings`, `allergy_warnings`, `changes_summary`, and `recommendations`, depending on the flow.

## RAG Pipeline

The RAG implementation is item-level RAG over structured database records, not long-document chunking.

1. Source data
   - Exercises are read from `exercises` joined with `general_exercises`.
   - Foods are read from `foods` joined with `general_nutrition`.
   - Database access uses PyMySQL and settings from `config.py`.

2. Document creation
   - Each exercise row becomes one Chroma document.
   - Each food row becomes one Chroma document.
   - The service intentionally avoids arbitrary character chunking.
   - Exercise documents include name, muscle group, body area, difficulty, instructions, mistakes, goal tags, search phrases, and inferred caution hints.
   - Food documents include name, category, serving size, calories, protein, carbs, fat, description, goal tags, and meal role tags.

3. Metadata and quality tags
   - Exercise metadata includes ID, type, name, raw and normalized muscle group, body area, difficulty, goal tags, and search quality.
   - Food metadata includes ID, type, name, category, normalized category, calories, macros, goal tags, meal role tags, and search quality.
   - Simple quality checks mark suspicious records as `junk` or `partial`; search excludes junk records by default.

4. Embeddings and vector storage
   - Embeddings are generated with `GoogleGenerativeAIEmbeddings`.
   - ChromaDB is created with `chromadb.PersistentClient`.
   - Collection names come from environment settings.
   - Vector IDs use stable prefixes such as `exercise:{id}` and `food:{id}`.

5. Sync and manifests
   - `/sync-all` runs exercise and nutrition sync.
   - Full sync batches records by `BATCH_SIZE`.
   - Incremental sync uses `data/sync_checkpoint.json`.
   - Manifest files store row fingerprints so unchanged rows do not need to be re-embedded.
   - Deleted database records are pruned from Chroma and the manifests.

6. Retrieval
   - User/search/plan requests are converted into retrieval queries.
   - The query is embedded and sent to the relevant Chroma collection.
   - When filters are used, the service over-fetches and then applies Python-side filtering.
   - Results are converted to source-aware response items with IDs, names, scores, previews, and metadata.

7. Prompting and generation
   - Retrieved exercise or food context is formatted into compact context blocks.
   - Prompt builders require JSON-only responses and instruct the model to use retrieved IDs.
   - Gemini calls include response schemas from `rag_schemas.py`.
   - OpenRouter calls use the same messages, but the schema argument is not enforced by the OpenRouter client.

8. Parsing and response conversion
   - LLM text is parsed into a JSON object.
   - Pydantic models validate the expected response shape.
   - Valid RAG responses are converted to the legacy response format expected by the FitMind backend.
   - Returned sources are filtered or merged so they correspond to retrieved records used in the result.

9. Fallbacks
   - If retrieval, LLM generation, JSON parsing, schema validation, or legacy conversion fails, endpoints return a deterministic fallback where implemented.
   - Fallback responses include `generation_mode` and `fallback_reason`.

## Safety and Personalization

The service uses request context to personalize results and reduce unsafe recommendations:

- Training plan generation uses level, goal, injuries, weak points, preferences, liked exercises, previous plans, and available training days.
- Training modification uses the current plan, user feedback, pain areas, modification text, injuries, and retrieved exercise alternatives.
- Shoulder-pain guardrails avoid overhead pressing, direct shoulder isolation, push-up/dip/bench-style stressors, and risky replacements unless the context is explicitly rehab-safe.
- Chat answers avoid diagnosis, advise users to stop painful movements, and recommend coach or clinician review for injury-specific issues.
- Chat does not directly edit plans; plan-change requests are redirected to a modification request flow.
- Nutrition generation and modification use preferences, allergies, disliked foods, liked foods, dietary preferences, meal count, target macros, and estimated calories.
- Nutrition guardrails remove or replace allergy-conflicting and disliked foods when safe retrieved replacements exist.
- Nutrition modification validates that added or replaced food source IDs come from retrieved records.
- Macro and calorie guardrails recalculate totals, add retrieved foods when totals are too low, and trim excessive protein when possible.

The service provides fitness and nutrition guidance, not medical diagnosis or clinical clearance.

## Installation and Setup

Prerequisites:

- Python 3.11 recommended.
- A reachable MySQL database with the FitMind exercise and nutrition tables used by `main.py`.
- A Gemini API key for embeddings. Search and sync require embeddings even if generation uses OpenRouter.
- Optional OpenRouter credentials if `LLM_PROVIDER=openrouter` is used for generation.

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create and configure your environment file:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env` with local database, LLM, and vector-store settings. Do not commit real `.env` values.

Run the service:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

The provided `.env.example` and Dockerfile use port `8001`. If `PORT` is not set and `main.py` is executed directly, `config.py` falls back to port `8080`.

After configuring the database and Gemini key, run an initial full sync:

```bash
curl -X POST "http://localhost:8001/sync-all?full_sync=true"
```

PowerShell alternative:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8001/sync-all?full_sync=true"
```

## Docker Setup

Build the image:

```bash
docker build -t fitmind-ai-service .
```

Run the service with environment variables from `.env`:

```bash
docker run --env-file .env -p 8001:8001 fitmind-ai-service
```

To persist Chroma data and sync checkpoints outside the container, mount the relevant directories:

```bash
docker run --env-file .env -p 8001:8001 \
  -v "$(pwd)/chroma_gym:/app/chroma_gym" \
  -v "$(pwd)/data:/app/data" \
  fitmind-ai-service
```

This repository contains a `Dockerfile`; it does not currently contain a `docker-compose.yml`.

## Environment Variables

Use `.env.example` as the template. The repository intentionally keeps real secrets out of documentation.

| Category | Variables |
| --- | --- |
| Application/server | `APP_NAME`, `DEBUG`, `HOST`, `PORT`, `CORS_ALLOW_ORIGINS` |
| MySQL database | `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, `DB_DATABASE` |
| LLM provider | `LLM_PROVIDER` |
| Gemini | `GEMINI_API_KEY`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_GENERATION_MODEL` |
| OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| Chroma/vector store | `CHROMA_PATH`, `EXERCISES_COLLECTION`, `NUTRITION_COLLECTION` |
| Sync files and batching | `EXERCISES_MANIFEST`, `NUTRITION_MANIFEST`, `CHECKPOINT_FILE`, `BATCH_SIZE`, `SYNC_DELAY_SECONDS` |
| Internal API token placeholder | `INTERNAL_AI_API_TOKEN` |

No backend URL variable is currently used by this service. The Laravel backend is expected to call the AI service over HTTP, while this service reads exercise and nutrition source data directly from MySQL.

## Integration With FitMind

In the FitMind architecture, the Laravel backend calls this FastAPI service for AI operations. The backend can:

- Trigger sync when exercise or food data changes.
- Request semantic exercise and food search results.
- Request generated training and nutrition plans.
- Submit current plans plus user feedback for plan modification.
- Forward member-facing chat requests to `/chat`.
- Use `/analyze-progress` to decide whether a plan should continue or be modified.

The service returns structured data suitable for backend storage and review workflows, including source references, warnings, fallback reasons, and plan versions. The web dashboard and mobile application are outside this repository; they interact with this service through the Laravel backend.

## Known Notes and Limitations

- `GET /health` is a shallow liveness check. It does not verify MySQL, ChromaDB, or LLM connectivity.
- Most logic lives in `main.py`; there are no separate router or service modules yet.
- `INTERNAL_AI_API_TOKEN` is loaded but not enforced in the current code, for backward compatibility with existing Laravel calls.
- OpenRouter generation support does not enforce the Gemini-style `responseSchema`; returned text still needs to be valid JSON for the parsers.
- Incremental nutrition sync expects timestamp columns on the `foods` table. Exercise sync has additional timestamp-column detection and fallback behavior.
- Full RAG behavior depends on a populated MySQL database, successful Chroma sync, and a valid Gemini API key for embeddings.
- Rule-based fallbacks are intentionally simpler than RAG/LLM responses but preserve a usable response shape and include `fallback_reason`.
- No automated test suite is present in this repository.
- Docker Compose is not included.

## Graduation Project Note

FitMind AI Service is part of the FitMind graduation project, an AI-powered gym management and fitness platform combining backend workflows, coach review, member-facing experiences, and retrieval-augmented fitness/nutrition intelligence.

## Authors

FitMind graduation project team.

## License

License not specified. Add a project license before public distribution.
