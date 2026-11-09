-- Fields the parser and chunker produce that 001 didn't have.
-- Defaults only fill rows that already exist; they're dropped so new rows must set them.

ALTER TABLE pages ADD COLUMN release text;  -- exact docs release, e.g. '16.15'; NULL if unknown

ALTER TABLE chunks ADD COLUMN section_key text NOT NULL DEFAULT '';
ALTER TABLE chunks ALTER COLUMN section_key DROP DEFAULT;  -- 'SQL-SELECT/description'
ALTER TABLE chunks ADD COLUMN ordinal integer NOT NULL DEFAULT 0;
ALTER TABLE chunks ALTER COLUMN ordinal DROP DEFAULT;      -- position within the section

COMMENT ON COLUMN pages.content_hash IS
    'sha256 over the page''s chunks (ids, texts, metadata) and the embedding model. '
    'Unchanged hash = nothing to do for this page.';
