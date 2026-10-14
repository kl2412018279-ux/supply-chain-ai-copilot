-- Runs once when the Postgres volume is first created.
-- The application also creates these objects idempotently on startup.
CREATE EXTENSION IF NOT EXISTS vector;

-- Structured sales history (loaded from data/raw/sample_sales.csv on first API start)
CREATE TABLE IF NOT EXISTS sales (
    date  DATE    NOT NULL,
    sku   TEXT    NOT NULL,
    sales NUMERIC NOT NULL CHECK (sales >= 0),
    PRIMARY KEY (sku, date)
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
