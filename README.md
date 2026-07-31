# AI Data Analysis

Ask questions about your data in plain English and get answers back as tables
and charts. Your question is translated into SQL by a language model, checked,
executed, and the results are rendered in the browser.

The repository holds two applications that share the same idea at different
scales:

| | What it queries | Model | Entry point |
|---|---|---|---|
| **Data Analyst Agent** (root) | A CSV you upload, loaded into in-memory SQLite | DeepSeek-R1-Distill-Llama-70B via Hugging Face Inference | `main.py` |
| **Text-to-SQL LLM** (`text2sql/`) | A Databricks SQL warehouse, with accounts and query history | OpenAI GPT-4o | `text2sql/app.py` |

Start with the root app: it runs on your own file and needs one API token.

---

## Data Analyst Agent

Upload a CSV, ask a question, get a result set. Nothing is uploaded anywhere
except the schema and a few sample rows, which are sent to the model as prompt
context.

### Setup

```bash
git clone https://github.com/SdSarthak/AI-Data-Analyst.git
cd AI-Data-Analyst

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # then edit .env
```

You need a Hugging Face token with inference access. Create one at
<https://huggingface.co/settings/tokens> and put it in `.env`:

```
HF_TOKEN=hf_your_token_here
```

`HF_PROVIDER` and `HF_MODEL` in `.env.example` select which inference provider
and model to route to. Any chat-completion model your provider serves will
work; the default is DeepSeek-R1-Distill-Llama-70B through Nebius.

### Run

```bash
streamlit run main.py
```

Open <http://localhost:8501>, upload a CSV in the sidebar, and ask away.

### Where the data comes from

There is no bundled dataset, by design: no real data is committed to this
repository. Bring any CSV. Public sets that work well for trying it out:

- [NYC Taxi trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Kaggle Superstore sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- [Our World in Data](https://ourworldindata.org/) exports

The file is read into a pandas DataFrame and registered as a SQLite table named
`data`. Ask questions about it by column name.

### Example questions

- "What is the average revenue by region?"
- "Show me the top 10 customers by total spend"
- "How many orders had more than 3 items?"
- "Group by category and show the sum of sales, highest first"

### How it works

```
CSV upload
   -> pandas DataFrame
   -> CREATE TABLE schema + 3 sample rows                 (analyst/schema.py)
   -> prompt -> model -> raw response                     (analyst/llm.py)
   -> strip reasoning blocks, fences and prose            (analyst/llm.py)
   -> read-only + single-statement checks, EXPLAIN        (analyst/sql.py)
   -> execute on a read-only in-memory SQLite connection  (analyst/sql.py)
   -> table, CSV download, chart                          (main.py)
```

Two things are worth calling out:

**Generated SQL is untrusted.** Every query is checked for write keywords and
stacked statements with string literals and comments masked out first, so
`WHERE city = 'Update Falls'` is not mistaken for an `UPDATE`. It is then
prepared with `EXPLAIN` to catch unknown columns before anything runs. On top
of that the SQLite connection installs an authorizer that permits only reads,
so a statement slipping past the text checks still fails at the driver.

**Results are capped.** A query with no `LIMIT` gets one appended
(`MAX_RESULT_ROWS`, default 5000) so an unbounded `SELECT *` over a large
upload cannot exhaust memory.

### Layout

```
main.py                Streamlit UI, no business logic
analyst/
  config.py            environment-driven settings
  schema.py            DataFrame -> CREATE TABLE DDL and sample rows
  llm.py               prompt building, model call, response cleaning
  sql.py               validation, read-only enforcement, execution
  errors.py            exception hierarchy
tests/                 pytest suite, synthetic fixtures only
text2sql/              the Databricks application (see text2sql/README.md)
```

All logic sits in `analyst/` rather than in the Streamlit script, so it can be
imported and tested without a browser.

---

## Text-to-SQL LLM (`text2sql/`)

A larger application that queries a Databricks SQL warehouse, with user
accounts, session history and saved queries. It needs Databricks credentials
and an OpenAI key. See [`text2sql/README.md`](text2sql/README.md) for setup.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

227 tests, all deterministic. Model calls are stubbed, fixtures are synthetic,
and nothing touches the network or a real database.

```bash
pytest tests/test_sql.py -v        # one module
pytest -k "read_only"              # one topic
```

## Configuration reference

Everything is read from the environment; see `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | – | Hugging Face token (required) |
| `HF_PROVIDER` | `nebius` | Inference provider |
| `HF_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Model id |
| `HF_MAX_TOKENS` | `500` | Response cap |
| `HF_TEMPERATURE` | `0.0` | Sampling temperature |
| `MAX_RESULT_ROWS` | `5000` | Row cap for unbounded queries |
| `PROMPT_SAMPLE_ROWS` | `3` | Example rows shown to the model |

`HUGGINGFACEHUB_API_TOKEN` and `HUGGINGFACE_API_KEY` are accepted as
alternatives to `HF_TOKEN`.

## Notes and limits

- Only `SELECT` is supported. This is a reading tool, not a writing one.
- Accuracy depends on clear column names. Questions that name columns
  explicitly do better than vague ones.
- The whole CSV is held in memory. Very large files are better sampled first.
- No dataset, credential or notebook output is committed here. `.gitignore`
  covers `*.csv`, `*.parquet`, `*.db`, `data/`, `.env` and friends.

## License

MIT
