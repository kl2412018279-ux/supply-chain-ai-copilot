-- Runs once when the Postgres volume is first created.
-- The application also creates these objects idempotently on startup.
CREATE EXTENSION IF NOT EXISTS vector;

-- Daily container volumes (loaded from data/raw/container_volumes.csv on first API start)
CREATE TABLE IF NOT EXISTS container_volumes (
    date      DATE    NOT NULL,
    commodity TEXT    NOT NULL,
    teu       NUMERIC NOT NULL CHECK (teu >= 0),
    PRIMARY KEY (commodity, date)
);

-- Document chunks: dense embedding (pgvector) + full-text index in one table
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page        INTEGER,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(384) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, chunk_index)
);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING gin (content_tsv);

-- No access through Supabase's public API (anon / authenticated keys); the app
-- connects as the table owner, which RLS does not restrict.
ALTER TABLE container_volumes ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
