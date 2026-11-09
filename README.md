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

## Parsing pages into sections

`docs_qa/parse.py` turns each saved page into sections of clean text blocks.

- **Only the documentation:** the page's `div#docContent` is kept; the site menu, version switcher, navigation, tables of contents and permalink icons are dropped.
- **Sections from the document structure:** sections come from the DocBook section elements, not from heading tags (note boxes have their own `<h3>Note</h3>`, which would otherwise become fake sections). Settings, libpq functions, psql commands and glossary terms have their own IDs (`GUC-AUTOVACUUM-NAPTIME`), so each becomes its own section.
- **Stable keys:** hand-written IDs (`SQL-WITH`) are used as keys. Generated IDs (`id-1.9.3.172.7`) encode a section's position in the book and change when pages are added (179 of 1,382 such sections moved between 16 and 17), so those sections are keyed by parent and heading (`SQL-SELECT/description`). Links still use the real ID.
- **Unsplittable blocks:** a section is a list of paragraphs, code examples, table rows, list items and definitions. The chunker splits only between blocks, so a function's signature, description and example stay together.

```bash
uv run docs-qa parse 16/sql-select.html   # show one page's sections and blocks
uv run docs-qa parse --all                # parse every page, print totals
```

| Version | Pages | Sections | Blocks | Pages with no content |
|---|---|---|---|---|
| 16 | 1,169 | 6,040 | 32,717 | 16 |
| 17 | 1,143 | 6,160 | 33,161 | 11 |
| 18 | 1,148 | 6,286 | 33,653 | 11 |

Pages with no content are tables of contents and the alphabetical index.

## Chunking

`docs_qa/chunk.py` packs each section's blocks into chunks for embedding and search. Sizes are measured with the embedding model's own tokenizer (`BAAI/bge-small-en-v1.5`, via the `tokenizers` library).

- **At most 400 tokens per chunk**, location line included. The embedding model accepts 512, but a cross-encoder reranker reads the question and the chunk together in one 512-token window; 400 leaves room for the question.
- **Splits only between blocks**, so a paragraph, code example, table row or setting is never cut in half. A block over the limit on its own is split between lines (code, with the fences repeated) or sentences (prose).
- **Balanced sizes**: the fewest chunks that fit, as equal as possible (60 + 60 + 60, not 80 + 80 + 20).
- **One section, its own chunks**: sections are never merged, so every chunk links to an exact anchor.
- **Location line without the version**: each chunk starts with e.g. `SELECT › Parameters › WITH Clause`. The version is a column used for filtering and citations, so text that is unchanged across versions has the same content hash and is embedded once.

Chunk IDs are `<version>/<page>#<section key>/<n>`, e.g. `16/sql-select.html#SQL-WITH/0`.

```bash
uv run docs-qa chunk 16/sql-select.html   # show one page's chunks
uv run docs-qa chunk --report             # chunk everything, print sizes and savings
```

| Version | Chunks | Tokens p50 | p90 | p99 | Max |
|---|---|---|---|---|---|
| 16 | 8,815 | 207 | 358 | 392 | 400 |
| 17 | 8,995 | 206 | 357 | 393 | 400 |
| 18 | 9,177 | 206 | 357 | 394 | 400 |

- 442 blocks were too big on their own: 222 code blocks split between lines, 220 prose blocks between sentences, none cut mid-sentence.
- 12,848 distinct chunk texts out of 26,987 chunks: 52% of chunks repeat text from another version and need no embedding of their own.
- Chunking all three versions takes about 46 seconds on a MacBook Air.

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
  parse.py                pages -> sections of text blocks
  chunk.py                sections -> chunks of at most 400 tokens
  tokens.py               token counting with the embedding model's tokenizer
  sql/                    numbered schema migrations
tests/
  fixtures/16/            trimmed real pages for the parser tests
```
