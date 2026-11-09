# grounded-docs-qa

Question answering over the PostgreSQL documentation (versions 16, 17 and 18) that only answers from retrieved passages and cites them. Built on [llm-lab](https://github.com/sunnypatel2048/llm-lab), my shared LLM client.

> Work in progress. Sections below fill in as each part is built.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/sunnypatel2048/grounded-docs-qa.git && cd grounded-docs-qa
uv sync
cp .env.example .env

docker compose up -d --wait     # pgvector on localhost:5433
uv run docs-qa db up            # create the schema
uv run pytest -q
```

## How it works

_Coming: ingestion, hybrid retrieval, reranking, citation enforcement, evaluation._

## Database

Schema changes are numbered SQL files in [`src/docs_qa/sql/`](src/docs_qa/sql/), applied in order by `docs-qa db up` and recorded in `schema_migrations`. An applied file is never edited: its checksum is stored, so a change has to go in a new file.

| Table | Holds |
|---|---|
| `pages` | One row per documentation page per version |
| `chunks` | The retrievable passages: text, location (version, page, section), token count, embedding |
| `embedding_cache` | Embeddings by content hash, so text repeated across versions is embedded once |

## Layout

```
docker-compose.yml        pgvector (PostgreSQL 18) on port 5433
src/docs_qa/
  cli.py                  the docs-qa command
  db.py                   connection and migration runner
  sql/                    numbered schema migrations
tests/
```
