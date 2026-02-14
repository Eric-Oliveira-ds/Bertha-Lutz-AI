CREATE TABLE memory (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
