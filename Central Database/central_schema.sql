-- Create the companies table
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

-- Create the divisions table
CREATE TABLE divisions (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id),
    name TEXT,
    tag TEXT,
    api_url TEXT,
    api_key VARCHAR(128) NOT NULL UNIQUE
);

-- Create the connections table
CREATE TABLE connections (
    id SERIAL PRIMARY KEY,
    source_division_id INT REFERENCES divisions(id),
    target_division_id INT REFERENCES divisions(id),
    daily_messages_count INT DEFAULT 0,
    api_key VARCHAR(128) NOT NULL UNIQUE
);

-- Create the collab_threads table
CREATE TABLE collab_threads (
    thread_id SERIAL PRIMARY KEY,
    connection_id INT REFERENCES connections(id),
    messages_count INT DEFAULT 0,
    last_message_timestamp TIMESTAMP,
    source_division_cost NUMERIC DEFAULT 0,
    target_division_cost NUMERIC DEFAULT 0
);

-- Create the llm_context_windows table
CREATE TABLE llm_context_windows (
    id SERIAL PRIMARY KEY,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    context_window_length INT NOT NULL,
);
