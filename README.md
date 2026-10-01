# Memory-Aware Assistant

A locally runnable FastAPI service that answers chat messages using recent
conversation and relevant user knowledge stored in Neo4j. Useful facts are
extracted after answer generation, validated, and saved for future sessions.

The core demonstration: a user plans to change jobs in 2027, the assistant recalls
that goal in a new session, the user corrects it to 2028, and another session
recalls the corrected goal. This is an API; no frontend or cloud deployment is required.

## Run locally

Prerequisites:

- Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Docker Desktop with its engine running, or Docker Engine with Compose.
- An OpenAI API key with API credits and access to `gpt-4.1-mini` (the default).
- Local ports 8000, 7474, and 7687 available.

Clone/download the repository and open a terminal in its root directory. Commands
below use a macOS/Linux shell. No separate Neo4j installation or account is needed.

```bash
# Install the locked application and development dependencies into .venv.
uv sync --locked

# Create private settings without replacing an existing .env file.
cp -n .env.example .env
```

Edit `.env` locally. Keep the real key and password out of source control:

```dotenv
OPENAI_API_KEY=your_actual_openai_key
OPENAI_MODEL=gpt-4.1-mini
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_DATABASE=neo4j
NEO4J_PASSWORD='choose_a_new_private_password'
```

Choose a new database password of at least 8 characters. The initial database
username is `neo4j`. Environment variables override values in `.env`.

```bash
# Validate Compose settings without printing credentials.
docker compose config --quiet

# Download the pinned Neo4j image if needed and start it in the background.
docker compose up -d

# After Neo4j finishes starting, verify credentials and database access from Python.
uv run python -m app.brain

# Start the API with automatic reload on Python-file changes.
uv run uvicorn app.main:app --reload
```

The first image download/startup takes time. The connection check should print
`Neo4j connection successful (connected=1).` If it fails during startup, wait and
retry. It does not create user data.

- Interactive API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- API liveness: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Neo4j Browser: [http://localhost:7474](http://localhost:7474)

In Neo4j Browser, connect to `bolt://localhost:7687` with the username and password
from `.env`. To check it without changing data, run `RETURN 1 AS connected;`.

Use one Uvicorn worker and sequential requests within a conversation for this demo.
Automatic code reload clears short-term history; persistent graph data survives.

## API and sample requests/responses

`POST /chat` accepts required nonblank `user_id`, `session_id`, and `message`
strings, plus an optional `profile`. Identifiers are limited to 128 characters
and the message to 4,000 characters. Profile fields are optional: name, date/time
of birth, birth place, preferred language, and zodiac/sun sign.

Use [examples/requests.http](examples/requests.http) in an IDE HTTP client, or paste
its JSON bodies into `/docs`. The same scenarios also have automated tests;
the example file is for manually operating the app. Change the example user ID
for a fresh run, since saved facts persist. When copying from the `.http` file
into `/docs`, replace `{{user}}` with `demo-review-1`; only an IDE HTTP client
expands that file's variables automatically.

### 1. Create a memory

This curl command posts a profile and goal to the running API:

```bash
curl -sS http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-review-1","session_id":"session-a","message":"I plan to change jobs in 2027.","profile":{"name":"Rahul"}}'
```

Illustrative response (model wording varies; sample wording is not an exact-match
acceptance criterion):

```json
{
  "response": "You plan to change jobs in 2027.",
  "user_id": "demo-review-1",
  "session_id": "session-a",
  "context_used": ["user_profile"],
  "memory_status": "saved",
  "warnings": []
}
```

### 2. Recall in a new session

Request:

```json
{
  "user_id": "demo-review-1",
  "session_id": "session-b",
  "message": "What are my career plans?"
}
```

Illustrative response:

```json
{
  "response": "You plan to change jobs in 2027.",
  "user_id": "demo-review-1",
  "session_id": "session-b",
  "context_used": ["user_profile", "career_goal"],
  "memory_status": "unchanged",
  "warnings": []
}
```

### 3. Correct the goal

Request:

```json
{
  "user_id": "demo-review-1",
  "session_id": "session-b",
  "message": "Actually, I plan to change jobs in 2028, not 2027."
}
```

Illustrative response:

```json
{
  "response": "Your updated plan is to change jobs in 2028.",
  "user_id": "demo-review-1",
  "session_id": "session-b",
  "context_used": ["recent_conversation", "user_profile", "career_goal"],
  "memory_status": "saved",
  "warnings": []
}
```

Repeat request 2 using `session-c`: the answer should mention **2028**, with
`career_goal` in `context_used` and no `recent_conversation` for this new session.
The stored key remains `career_change`; correcting its year updates the same fact.

`context_used` describes the categories supplied to the answer model, not proof
that every supplied field influenced its wording. `memory_status` means:

| Value | Meaning |
| --- | --- |
| `saved` | A profile/fact update was committed. Check warnings for partial extraction failure. |
| `unchanged` | Extraction succeeded but no new durable information needed saving. |
| `unavailable` | Persistent storage could not be used; basic chat used available current/session context. |
| `failed` | The answer succeeded, but extraction or the subsequent write failed. |

Warnings are explicit: an HTTP 200 answer alone is not evidence that memory was
saved. When extraction fails but an explicitly supplied profile can be saved,
`saved` may accompany a warning about unextracted facts. Profile changes and
validated memory facts submitted together are committed atomically.

## Architecture and flow

```text
POST /chat -> validate request -> load recent user/session history
          -> select topics -> read relevant Neo4j facts and profile
          -> assemble context -> OpenAI generates answer
          -> OpenAI extracts candidate updates from the user's message
          -> validate evidence/fields -> save updates in Neo4j
          -> append successful exchange to RAM history -> return response
```

Answer generation happens before memory extraction. Both finish before the HTTP
response returns, so a following request can read a successful update. This
avoids background-write races at the cost of additional response latency.

| Module | Responsibility / main entry points |
| --- | --- |
| `app/main.py` | FastAPI routes, driver lifecycle, dependency injection, HTTP error mapping. |
| `app/models.py` | Request/response, profile, fact, and extraction validation schemas. |
| `app/config.py` | Load and validate separate OpenAI/Neo4j settings; redact secrets in settings representations. |
| `app/chat.py` | `handle_chat()` coordinates retrieval, generation, updates, and warnings. |
| `app/context.py` | `select_topics()`, `merge_profile()`, `select_profile()` implement transparent selection rules. |
| `app/memory.py` | `ConversationMemory` keeps recent exchanges separately for each user/session. |
| `app/brain.py` | `BrainStore` reads/writes the graph; command-line connectivity/storage demo. |
| `app/llm.py` | `generate_reply()`, `extract_updates()`, `validate_extraction()` isolate provider calls. |
| `tests/` | Offline behavior/provider tests and opt-in real Neo4j tests. |
| `compose.yaml` | Local Neo4j Community container and named data volume. |

The provider can be replaced by implementing the same generation/extraction
interfaces in `llm.py`; callers use application models rather than SDK objects.
Python functions are sufficient for this fixed workflow; no agent framework is needed.

## Shared Brain schema

```text
(:User {user_id, name?, date_of_birth?, time_of_birth?, birth_place?,
        preferred_language?, zodiac_sign?, created_at, updated_at?})
    |
    +--[:HAS_MEMORY]--> (:Memory {user_id, key, kind, topic, content,
                                target_year?, timeframe?, created_at, updated_at})
```

- One `User` node per unique `user_id`; profile fields live on that node.
- One `Memory` node per unique `(user_id, key)`. The owner is also recorded on
  the node so retrieval can verify the user scope alongside the relationship.
- `kind` distinguishes `goal`, `preference`, `interest`, `important_memory`,
  `life_area`, and `astrology`. A generic relationship keeps writes simple while
  these properties carry meaning; typed relations/multi-hop entities can be added later.
- `MERGE` reuses the same user/fact, and `SET` updates its properties. Distinct
  goals need distinct keys; an explicit correction reuses its existing key.
- Profile writes patch supplied fields only. An explicitly supplied null clears
  a field; omitted fields remain unchanged. Dates/times are stored as ISO strings.
- Parameterized Cypher avoids interpolating user text. Uniqueness constraints
  prevent duplicate user/fact nodes. Each chat update uses one write transaction.
- Graph data lives in Docker's `neo4j_data` named volume, separate from RAM and
  the container lifecycle. "Shared" means reusable across that user's sessions,
  not shared between users.

Neo4j matches the graph-based requirement and makes ownership and future
relationships explicit. Current retrieval is deliberately one-hop and filtered;
the design does not claim advanced graph reasoning is necessary for this demo.

## Memory and context selection

**Short-term:** up to five complete user/assistant exchanges per `(user_id,
session_id)`, across at most 100 sessions. The least recently used session is
evicted at capacity. Restarts/reloads clear all session history. The limits bound
message/session counts rather than specifying a fixed RAM allocation.

**Long-term:** an LLM proposes only useful information explicitly stated by the
user. Greetings, pure questions, hypothetical plans, and assistant suggestions
should produce no updates. At most five facts and six profile-field updates are
accepted per turn. Each proposal must include an exact, nonblank quote from the
current message. Pydantic validates fields; duplicate keys/fields are rejected.
Evidence matching does not prove semantic truth: extraction remains fallible.

Relative years use the host's current date. Other relative timeframes retain
their reference date. Corrections use stable keys, with `career_change` explicitly
defined for the demo goal. A cancellation updates the fact's content rather than
deleting historical nodes. Advanced conflict resolution/version history is deferred.

**Selection before generation:**

1. Match English topic keywords (career, relationships, family, health, finance,
   education, hobbies, astrology, preferences).
2. For a reference-style follow-up with no explicit topic, inspect recent user
   messages for a topic. An explicit new subject takes precedence.
3. Retrieve at most five matching memories inside Neo4j, ordered by update time.
   Broad "what do you remember about me?" requests retrieve a bounded overview.
   Unrecognized subjects retrieve no facts rather than dumping the graph.
4. Include available name/language and relevant profile details. Birth information
   is included for astrology/profile questions, not ordinary career questions.
5. Supply this selected data, up to five recent exchanges, and the current message.

This is a rules-plus-LLM hybrid: explainable rules for retrieval, structured LLM
output for extraction. No embeddings or semantic search are used. Keywords can
miss paraphrases; a generic follow-up can inherit an imperfect topic. Retrieval
is capped, so an older fact may not be included. These are documented trade-offs.

## Testing and evaluation

```bash
# Offline tests: no OpenAI credentials or database required; DB tests are skipped.
uv run pytest -q

# All tests, including local Neo4j; requires Docker DB and .env credentials.
RUN_NEO4J_TESTS=1 uv run pytest -q
```

The suite contains 44 test cases: 36 offline and eight opt-in database checks.
All 44 tests passed during local verification. The real 2027 -> 2028 cross-session
demo also passed; after the final prompt refinement and an API restart, a new
session recalled 2028 directly with no warnings. Re-run the commands above and
the manual demo in your own environment. Automated tests fake OpenAI calls;
they do not guarantee live model wording or extraction quality.

Database tests create unique test users and clean up only those users' data.
Schema constraints remain. They never call OpenAI or delete demo/user data.

The requested behavior is organized into these ten scenarios; individual tests
also cover variations and boundaries:

| Scenario | Expected behavior | Coverage |
| --- | --- | --- |
| New user | Answer without invented profile/facts | `test_chat.py`, `test_brain.py` |
| Create durable memory | Valid extracted goal is saved | `test_chat.py`, `test_brain.py` |
| Retrieve stored memory | Relevant goal enters the answer context | `test_chat.py`, `test_brain.py` |
| Follow-up question | Recent exchange resolves the reference | `test_memory.py`, `test_context.py`, `test_chat.py` |
| New session / other user | Recall for the same user; no cross-user leakage | `test_chat.py`, `test_memory.py`, `test_brain.py` |
| Irrelevant memory | Unrelated facts are excluded | `test_context.py`, `test_chat.py`, `test_brain.py` |
| User correction | Replace the existing fact without duplication | `test_chat.py`, `test_brain.py` |
| Missing profile / invalid input | Optional fields work; invalid input gets 422 | `test_chat.py`, `test_brain.py` |
| Provider/database failure | Honest status; no writes on answer failure | `test_chat.py`, `test_llm.py`, `test_memory.py` |
| Persistence / atomicity | New connection reads saved facts; failed write rolls back | `test_brain.py` |

**Basic quality evaluation (proposed manual comparison, not a claimed benchmark):**
Use fixed synthetic conversations with expected facts and run the same questions
with and without stored-memory context, holding the model, prompt, and recent
history constant. Inspect both the extracted facts and resulting answers:

| Area | Measure |
| --- | --- |
| Memory accuracy | Correct extracted facts / all saved facts; expected facts captured / all expected facts. |
| Context relevance | Relevant retrieved facts / all retrieved facts; track missed required facts too. |
| Personalization | Score 0–2: generic, partially relevant, or appropriately uses known goals/preferences. |
| Consistency | Corrected facts replace stale claims; follow-ups refer to the right exchange. |
| Irrelevant context | Count unrelated facts injected or mentioned, including any other-user facts. |
| Persistence | Correct recall across new sessions, Python restarts, and database restarts. |

Repeat representative live runs because model behavior varies. Compare quality
alongside latency, provider-call count, and token usage; this repo does not include
a sophisticated evaluation framework or measured quality scores.

## Failure behavior and production considerations

| Failure | Behavior |
| --- | --- |
| Invalid JSON, blank/oversized input, invalid profile | HTTP 422 before generation. |
| Missing OpenAI settings | HTTP 503. |
| Answer-generation timeout | HTTP 504; neither memory store is updated. |
| Provider failure / empty or unfinished answer | HTTP 502; neither memory store is updated. |
| Missing profile / empty or irrelevant memories | Answer from available context; do not invent missing details. |
| Database unavailable | Answer from supplied profile/session context; `unavailable` status and warning. |
| Post-answer extraction or write failure | Preserve answer and short-term exchange; report failure in status/warnings. |

The app owns one Neo4j driver and closes it on shutdown. Schema setup retries on
subsequent requests if the database initially fails. If DB configuration was
missing at app startup, restart Uvicorn after supplying it.

This is a local demonstrator, not a production deployment:

- Caller-supplied IDs separate data but do not authenticate users. Production
  must derive user identity from authentication and authorize conversation access.
- RAM history is process-local. A production service needs persistent message
  storage/shared caching and retention policies. A lock protects individual RAM
  operations, but whole same-session turns are not serialized: send them sequentially.
- Requests can make two sequential model calls (answer and extraction), each with
  a 30-second SDK timeout and no automatic retries. Database outages add latency.
  Timeouts are SDK operation limits, not a strict end-to-end response deadline.
- Production needs bounded retries, request idempotency, per-user rate limits,
  monitoring, backups, encryption, access controls, and user deletion/export flows.
- Input size, history, output, and fact-count limits bound context, but there is no
  exact token-budget optimizer. The graph itself has no automatic expiry/size cap.
- Stored context is identified as untrusted data in prompts. Evidence validation
  reduces unsupported writes but does not eliminate prompt injection or semantic
  extraction mistakes. Secrets and sensitive data require stronger policies in production.
- OpenAI requests use `store=False`; this is not a blanket promise of zero provider
  data retention. API data handling still depends on provider policy/account controls.
- Corrections depend on appropriate fact keys and retrieved context. This is basic
  explicit replacement, not arbitrary contradiction detection or identity resolution.
- Zodiac can be supplied and used as profile data. Birth-chart calculations are
  intentionally not implemented; the assistant must not claim to calculate charts.

### Deferred features

Optional: memory importance/confidence scores, advanced conflict resolution,
expiration/decay, conversation summarization, model fallback/routing, advanced
token optimization, multilingual retrieval/evaluation, and advanced graph traversal.
Preferred language is stored and supplied to the model, but multilingual behavior
has not been systematically evaluated. UI, authentication, persistent full chat
history, cloud deployment, and semantic/vector search are also outside this scope.

## Persistence demonstration and troubleshooting

For a database-only demo, these commands use the reserved sample user
`storage-demo-rahul`; no OpenAI calls are made:

```bash
# Create/reset the sample profile and its 2027 career goal.
uv run python -m app.brain save-demo
# Read them from a new Python process.
uv run python -m app.brain read-demo
# Correct the same goal to 2028.
uv run python -m app.brain update-demo
# Confirm the new year and a memory count of one.
uv run python -m app.brain read-demo
```

To check persistence through a database restart:

```bash
# Restart only the Neo4j container; its data volume is retained.
docker compose restart neo4j
# Once startup finishes, read the same stored demo data.
uv run python -m app.brain read-demo
```

To inspect the API example's graph in Neo4j Browser, run this read-only query:

```cypher
MATCH (u:User {user_id: 'demo-review-1'})
OPTIONAL MATCH (u)-[:HAS_MEMORY]->(m:Memory)
RETURN u, m;
```

If Docker cannot connect to its daemon, start Docker Desktop. For startup trouble:

```bash
# Inspect container status and recent database startup logs.
docker compose ps
docker compose logs --tail=50 neo4j
```

Neo4j initializes its password only when its data volume is empty; editing `.env`
later does not change an existing database password. A JSON trailing comma causes
422 before application logic. On model access/authentication errors, check the
key, credits, and `OPENAI_MODEL` locally; do not share secrets in diagnostic output.

To stop the API, press Ctrl+C in its terminal. To stop Neo4j while keeping data:

```bash
docker compose stop
```

`docker compose up -d` starts it again. Normal container removal with
`docker compose down` preserves the named volume; adding `--volumes` deletes
the volume and its saved memories. Do not use that option when preserving data.

## Submission contents

The repository includes runnable source, locked dependencies, local database setup,
automated tests, architecture/schema/memory explanations, evaluation and production
notes, and sample API requests/responses. A GitHub repository or zip can contain
these files. Exclude `.env`, `.venv`, caches, local database files, and the original
assignment document. A reviewer needs their own OpenAI API credentials and a new
local Neo4j password; existing local volumes and secrets are not part of submission.
