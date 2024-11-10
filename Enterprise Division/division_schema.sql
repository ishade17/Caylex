-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Creates a table for connections
CREATE TABLE IF NOT EXISTS connections (
    local_connection_id SERIAL PRIMARY KEY,
    central_connection_id INT,  -- Added field to map to central database
    source_division_id UUID NOT NULL,
    target_division_id UUID NOT NULL,
    daily_messages_count INT,
    raw_api_key TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Creates a table to log messages
CREATE TABLE IF NOT EXISTS messages (
    message_id SERIAL PRIMARY KEY,
    connection_id INT,     
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
CREATE TABLE IF NOT EXISTS custom_data_policies (
    id SERIAL PRIMARY KEY,
    connection_id INT,
    confidentiality VARCHAR,
    data_type VARCHAR,
    natural_language_explanation TEXT,
    FOREIGN KEY (connection_id) REFERENCES connections(local_connection_id)
);

-- Creates a table for the policy documents held in storage
CREATE TABLE IF NOT EXISTS data_policy_doc_infos (
    document_id SERIAL PRIMARY KEY,
    file_name TEXT
);

-- Creates a table to link data policy documents to connections
CREATE TABLE IF NOT EXISTS document_connections (
    document_id INT,
    connection_id INT,
    PRIMARY KEY (document_id, connection_id),
    FOREIGN KEY (document_id) REFERENCES data_policy_doc_infos(document_id),
    FOREIGN KEY (connection_id) REFERENCES connections(local_connection_id)
);

-- Update data_policy_doc_chunks table
CREATE TABLE IF NOT EXISTS data_policy_doc_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    document_id INT,
    chunk_order INT,
    FOREIGN KEY (document_id) REFERENCES data_policy_doc_infos(document_id)
);

-- Create an ivfflat index on the embedding vector column for faster cosine similarity searches
CREATE INDEX IF NOT EXISTS data_policy_docs_embedding_idx
    ON data_policy_doc_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Create a single collab_threads table
CREATE TABLE IF NOT EXISTS collab_threads (
    thread_id SERIAL PRIMARY KEY,
    connection_id INT,
    messages_count INT,
    last_message_timestamp TIMESTAMP,
    source_division_cost NUMERIC,
    target_division_cost NUMERIC,
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