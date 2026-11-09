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
uv run docs-qa fetch            # download the docs (~1 hour, resumable)
uv run pytest -q
```

## How it works

_Coming: ingestion, hybrid retrieval, reranking, citation enforcement, evaluation._

## Getting the documentation

`docs-qa fetch` downloads the official HTML pages of the PostgreSQL 16, 17 and 18 docs from postgresql.org into `data/raw/<version>/` (git-ignored). It starts at each version's `index.html` and follows links that stay inside that version.

```bash
uv run docs-qa fetch --limit 20               # trial: 20 pages per version
uv run docs-qa fetch                          # everything (16, 17, 18)
uv run docs-qa fetch --versions 18 --delay 2  # one version, slower
```

- **Polite:** reads robots.txt first and skips anything it disallows, waits between requests (1s by default, longer if robots.txt asks), identifies itself with a User-Agent that links to this repo, and backs off on 429 and 5xx (honouring `Retry-After`, capped at 60s). It stops after 10 failures in a row rather than keep hitting a struggling server.
- **Resumable:** a page already on disk is never downloaded again, so Ctrl-C and re-running the same command continues where it stopped, and a complete re-run makes no page requests. Pages are written atomically (temp file, then rename), so a crash can't leave a half-written page.
- **Recorded:** each version's `manifest.jsonl` holds one line per page: URL, size, SHA-256 and download time.

## Database

Schema changes are numbered SQL files in [`src/docs_qa/sql/`](src/docs_qa/sql/), applied in order by `docs-qa db up` and recorded in `schema_migrations`. An applied file is never edited: its checksum is stored, so a change has to go in a new file.

| Table | Holds |
|---|---|
| `pages` | One row per documentation page per version |
| `chunks` | The retrievable passages: text, location (version, page, section), token count, embedding |
| `embedding_cache` | Embeddings by content hash, so text repeated across versions is embedded once |

## Layout

```
data/raw/<version>/       downloaded HTML pages + manifest.jsonl (git-ignored)
docker-compose.yml        pgvector (PostgreSQL 18) on port 5433
src/docs_qa/
  cli.py                  the docs-qa command
  db.py                   connection and migration runner
  fetch.py                polite, resumable docs downloader
  sql/                    numbered schema migrations
tests/
```
