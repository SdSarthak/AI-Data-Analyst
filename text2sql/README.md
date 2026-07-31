# Text-to-SQL LLM

A Streamlit application that turns plain English questions into SQL, runs them
against a Databricks SQL warehouse, and shows the results. It adds user
accounts, per-session query history and saved queries on top of the pipeline.

This is a subproject of [AI Data Analysis](../README.md). If you just want to
query a CSV, use the root application instead — it needs one API token and no
warehouse.

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key
- A Databricks SQL warehouse, and from it:
  - the workspace host name
  - the warehouse HTTP path
  - a personal access token
  - the catalog and schema you want to query

## Setup

```bash
cd text2sql

python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # then edit .env
```

`setup.sh` (macOS/Linux) and `setup.bat` (Windows) do the same three steps.

### Configuration

All settings come from the environment or a local `.env`. Nothing is hardcoded.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | – | OpenAI credentials |
| `OPENAI_MODEL` | no | `gpt-4o` | Chat model used for generation |
| `OPENAI_TEMPERATURE` | no | `0.0` | Deterministic by default |
| `OPENAI_MAX_TOKENS` | no | `2000` | Response cap |
| `DATABRICKS_HOST` | yes | – | Workspace host; a full URL is trimmed |
| `DATABRICKS_HTTP_PATH` | yes | – | e.g. `/sql/1.0/warehouses/abc123` |
| `DATABRICKS_TOKEN` | yes | – | Personal access token |
| `DATABRICKS_CATALOG` | yes | – | Catalog to query |
| `DATABRICKS_SCHEMA` | yes | – | Schema to query |
| `USER_STORE_PATH` | no | `.data/users.json` | Where accounts are persisted |
| `SESSION_TIMEOUT_MINUTES` | no | `30` | Idle timeout |
| `MAX_QUERY_LENGTH` | no | `1000` | Longest accepted question |
| `MAX_ROWS_DISPLAY` | no | `100` | Rows fetched per query |
| `ENABLE_QUERY_OPTIMIZATION` | no | `True` | Reformat SQL before running |

Missing required settings are reported on the login screen rather than
crashing the app.

## Run

```bash
cd text2sql
streamlit run app.py
```

Open <http://localhost:8501>, create an account, and start asking questions.
Run it from inside `text2sql/` so the `src`, `config` and `utils` packages
resolve.

### Docker

```bash
cd text2sql
docker compose up --build
```

Accounts live in the `user-store` named volume so they survive a rebuild. The
nginx reverse proxy is behind a profile and is off by default; start it with
`docker compose --profile proxy up` once `nginx.conf` and `./certs` are in
place. See [SETUP.md](SETUP.md) for EC2 deployment.

## How it works

```
question
  -> schema + column-level table definitions read from Databricks
  -> prompt -> GPT-4o -> raw response
  -> fences, reasoning blocks and prose stripped
  -> validated: single statement, read-only, balanced
  -> reformatted onto clause-per-line
  -> executed with a row cap
  -> results, optional plain-English explanation
```

Schema and table definitions are fetched once per session and cached; call
`QueryEngine.refresh_schema()` after a DDL change.

### Safety

- **Read-only.** A query must be a single statement starting with `SELECT` or
  a CTE. Write keywords (`DROP`, `DELETE`, `UPDATE`, `MERGE`, `GRANT`, …) are
  refused.
- **Literal-aware checks.** Keyword matching runs against a copy of the query
  with string literals and comments blanked out, so a value like
  `'update me'` does not trip the filter, and a write hidden in a comment
  cannot smuggle itself past it.
- **No stacked statements.** `SELECT 1; DROP TABLE x` is rejected.
- **Identifiers are validated.** Catalog, schema and table names are
  interpolated into SQL, so they must match `[A-Za-z_][A-Za-z0-9_]*`.
- **Favourites are re-validated** before they run. Stored text is not trusted
  because it was valid when it was saved.
- **Passwords** are bcrypt hashed with a per-user salt at cost 12. Login
  returns the same message for an unknown user and a wrong password.

## Layout

```
app.py                    Streamlit UI
config/settings.py        environment-driven settings
src/
  llm_engine.py           prompt building, model call, response cleaning
  database_connector.py   Databricks connection, schema reads, execution
  sql_validator.py        safety checks, formatting, query introspection
  query_engine.py         orchestrates the pipeline
  auth.py                 accounts (persisted) and sessions (in memory)
utils/
  logger.py, errors.py
data/schema_examples.sql  example warehouse schema this app was built against
Dockerfile, docker-compose.yml, nginx.conf, deployment/
```

`src/llm_engine.py` and `src/database_connector.py` import LangChain and the
Databricks driver lazily, so the modules stay importable — and testable — on a
machine that has neither installed. Both `TextToSQLLLM` and `DatabaseConnector`
accept an injected client, which is how the test suite exercises them.

## Tests

The suite lives at the repository root and covers this subproject:

```bash
cd ..
pip install -r requirements-dev.txt
pytest -k text2sql
```

No network, no warehouse and no OpenAI key required.

## Known limits

- Accounts are stored in a local JSON file. That is fine for a single instance;
  a multi-instance deployment needs a real user database.
- Sessions live in the process, so a restart logs everyone out.
- The model sees every table definition in the schema. On a very wide schema
  this will need narrowing to the relevant tables.
- `SELECT`-only. Writing to the warehouse is out of scope.
