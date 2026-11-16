-- Private conversation metadata and LangGraph checkpoints in the existing
-- Supabase Postgres database. Layout follows langgraph-checkpoint-postgres 3.2.1.
CREATE SCHEMA IF NOT EXISTS legal_chat;

REVOKE ALL ON SCHEMA legal_chat FROM PUBLIC, anon, authenticated, service_role;

CREATE TABLE IF NOT EXISTS legal_chat.conversations (
    id uuid PRIMARY KEY,
    title text NOT NULL CHECK (length(btrim(title)) BETWEEN 1 AND 160),
    document_scope jsonb NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(document_scope) = 'array'),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE legal_chat.conversations ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS conversations_updated_at_idx
    ON legal_chat.conversations (updated_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS legal_chat.checkpoint_migrations (v integer PRIMARY KEY);
CREATE TABLE IF NOT EXISTS legal_chat.checkpoints (
    thread_id text NOT NULL,
    checkpoint_ns text NOT NULL DEFAULT '',
    checkpoint_id text NOT NULL,
    parent_checkpoint_id text,
    type text,
    checkpoint jsonb NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS legal_chat.checkpoint_blobs (
    thread_id text NOT NULL,
    checkpoint_ns text NOT NULL DEFAULT '',
    channel text NOT NULL,
    version text NOT NULL,
    type text NOT NULL,
    blob bytea,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
CREATE TABLE IF NOT EXISTS legal_chat.checkpoint_writes (
    thread_id text NOT NULL,
    checkpoint_ns text NOT NULL DEFAULT '',
    checkpoint_id text NOT NULL,
    task_id text NOT NULL,
    idx integer NOT NULL,
    channel text NOT NULL,
    type text,
    blob bytea NOT NULL,
    task_path text NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
ALTER TABLE legal_chat.checkpoint_blobs ALTER COLUMN blob DROP NOT NULL;
ALTER TABLE legal_chat.checkpoint_writes ADD COLUMN IF NOT EXISTS task_path text NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON legal_chat.checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON legal_chat.checkpoint_blobs(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON legal_chat.checkpoint_writes(thread_id);
INSERT INTO legal_chat.checkpoint_migrations(v)
VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)
ON CONFLICT (v) DO NOTHING;

ALTER TABLE legal_chat.checkpoint_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_chat.checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_chat.checkpoint_blobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_chat.checkpoint_writes ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON ALL TABLES IN SCHEMA legal_chat FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA legal_chat FROM PUBLIC, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA legal_chat
    REVOKE ALL ON TABLES FROM PUBLIC, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA legal_chat
    REVOKE ALL ON SEQUENCES FROM PUBLIC, anon, authenticated, service_role;
