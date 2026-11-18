-- pgvector must exist before 001_schema.sql creates vector(1024) columns.
CREATE EXTENSION IF NOT EXISTS vector;
