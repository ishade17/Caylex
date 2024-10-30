-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Creates a table for connections
CREATE TABLE IF NOT EXISTS connections (
    local_connection_id SERIAL PRIMARY KEY,
    source_division_id UUID NOT NULL,
    target_division_id UUID NOT NULL,
    daily_messages_count INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Creates a table to log messages
CREATE TABLE IF NOT EXISTS messages (
    message_id SERIAL PRIMARY KEY,
    connection_id INT,  -- Add this line    
    sender_division_id INT,
    receiver_division_id INT,
    message_content TEXT,
    timestamp TIMESTAMP,
    status VARCHAR,
    thread_id INT,
    thread_msg_ordering INT,
    token_counts JSONB,
    FOREIGN KEY (connection_id) REFERENCES connections(local_connection_id)
);

-- Creates a table for custom data policies
CREATE TABLE IF NOT EXISTS data_policies (
    id SERIAL PRIMARY KEY,
    connection_id INT,  -- Add this line
    confidentiality VARCHAR,
    data_type VARCHAR,
    natural_language_explanation TEXT,
    FOREIGN KEY (connection_id) REFERENCES connections(local_connection_id)
);

-- Creates a table for data policy documents
CREATE TABLE IF NOT EXISTS data_policy_docs (
    id SERIAL PRIMARY KEY,
    data_type TEXT,
    content TEXT,
    embedding vector(1536),
    document_num INT,
    chunk_order INT
);

-- Create an ivfflat index on the embedding vector column for faster cosine similarity searches
CREATE INDEX IF NOT EXISTS data_policy_docs_embedding_idx
    ON data_policy_docs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Creates a table to link data policy documents to connections
CREATE TABLE IF NOT EXISTS document_connections (
    document_chunk_id INT,
    connection_id INT,
    PRIMARY KEY (document_chunk_id, connection_id),
    FOREIGN KEY (document_chunk_id) REFERENCES data_policy_docs(id),
    FOREIGN KEY (connection_id) REFERENCES connections(local_connection_id)
);

-- Create the updated function with the new return type
CREATE OR REPLACE FUNCTION match_document_chunks(
  table_name TEXT,
  document_chunk_ids BIGINT[],
  query_embedding vector(1536),
  match_threshold FLOAT,
  match_count INT
) RETURNS TABLE(
  id BIGINT,
  content TEXT,
  document_num INT,
  chunk_order INT,
  similarity FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY EXECUTE format('
    SELECT
      id,
      content,
      document_num,
      chunk_order,
      1 - (embedding <=> $1) AS similarity
    FROM
      %I
    WHERE
      id = ANY($2) AND
      (1 - (embedding <=> $1)) > $3
    ORDER BY
      embedding <=> $1
    LIMIT $4
  ', table_name) USING query_embedding, document_chunk_ids, match_threshold, match_count;
END;
$$;




-- -- Creates a table to log messages
-- CREATE TABLE IF NOT EXISTS messages (
--     message_id SERIAL PRIMARY KEY,
--     connection_id INT,
--     sender_division_id INT,
--     receiver_division_id INT,
--     message_content TEXT,
--     timestamp TIMESTAMP,
--     status VARCHAR,
--     thread_id INT,
--     thread_msg_ordering INT,
--     token_counts JSONB
-- );

-- -- Creates a table for custom data policies
-- CREATE TABLE IF NOT EXISTS  data_policies (
--     id SERIAL PRIMARY KEY,
--     connection_id INT,
--     confidentiality VARCHAR,
--     data_type VARCHAR,
--     natural_language_explanation TEXT
-- );

-- -- Install the required extension; this needs to be done once per database.
-- CREATE EXTENSION IF NOT EXISTS pg_ivfflat;

-- -- Creates a table for data policy documents
-- CREATE TABLE IF NOT EXISTS data_policy_docs (
--     id SERIAL PRIMARY KEY,
--     data_type TEXT,
--     content TEXT,
--     embedding VECTOR(1536),
--     document_num INT,
--     chunk_order INT
-- );

-- -- Create an ivfflat index on the embedding vector column for faster cosine similarity searches
-- CREATE INDEX ON data_policy_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- -- Creates a table to link data policy documents to connections
-- CREATE TABLE IF NOT EXISTS  document_connections (
--     document_chunk_id INT,
--     connection_id INT,
--     PRIMARY KEY (document_chunk_id, connection_id),
--     FOREIGN KEY (document_chunk_id) REFERENCES data_policy_docs(id),
--     FOREIGN KEY (connection_id) REFERENCES connections(id)
-- );

-- -- Create the updated function with the new return type
-- CREATE OR REPLACE FUNCTION match_document_chunks(
--   table_name TEXT,
--   document_chunk_ids BIGINT[],
--   query_embedding VECTOR(1536),
--   match_threshold FLOAT,
--   match_count INT
-- ) RETURNS TABLE(
--   id BIGINT,
--   content TEXT,
--   document_num INT,
--   chunk_order INT,
--   similarity FLOAT
-- ) LANGUAGE plpgsql AS $$
-- BEGIN
--   RETURN QUERY EXECUTE format('
--     SELECT
--       id,
--       content,
--       document_num,
--       chunk_order,
--       1 - (embedding <=> $1) AS similarity
--     FROM
--       %I
--     WHERE
--       id = ANY($2) AND
--       embedding <=> $1 < 1 - $3
--     ORDER BY
--       embedding <=> $1
--     LIMIT $4
--   ', table_name) USING query_embedding, document_chunk_ids, match_threshold, match_count;
-- END;
-- $$;
