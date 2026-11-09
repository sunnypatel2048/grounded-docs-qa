-- Initial schema: downloaded pages, their chunks, and an embedding cache.
-- vector(384) matches BAAI/bge-small-en-v1.5. A different model size means a new migration.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per documentation page per version.
CREATE TABLE pages (
    version      text        NOT NULL,             -- '16', '17', '18'
    path         text        NOT NULL,             -- 'runtime-config-autovacuum.html'
    url          text        NOT NULL,
    title        text        NOT NULL,
    content_hash text        NOT NULL,             -- sha256 of the parsed page text
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (version, path)
);

-- The retrievable units. IDs come from where a chunk sits in the docs, so re-running
-- ingestion produces the same IDs and golden-set labels stay valid.
CREATE TABLE chunks (
    id              text        PRIMARY KEY,       -- '16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME/0'
    version         text        NOT NULL,
    page_path       text        NOT NULL,
    anchor          text,                          -- section id within the page; NULL = top of page
    section_path    text[]      NOT NULL,          -- {'Server Configuration','Automatic Vacuuming',...}
    url             text        NOT NULL,          -- page URL plus #anchor: what a citation links to
    content         text        NOT NULL,          -- embedded and shown to the LLM, location header included
    content_hash    text        NOT NULL,          -- sha256 of content: unchanged chunks are skipped
    token_count     integer     NOT NULL,          -- in the embedding model's tokenizer
    embedding       vector(384),                   -- NULL until embedded
    embedding_model text,
    updated_at      timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (version, page_path) REFERENCES pages (version, path) ON DELETE CASCADE
);

CREATE INDEX chunks_version_idx ON chunks (version);
CREATE INDEX chunks_page_idx ON chunks (version, page_path);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Text identical across versions (most pages barely change between 16, 17 and 18) is
-- embedded once and reused by hash.
CREATE TABLE embedding_cache (
    model        text        NOT NULL,
    content_hash text        NOT NULL,
    embedding    vector(384) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, content_hash)
);
