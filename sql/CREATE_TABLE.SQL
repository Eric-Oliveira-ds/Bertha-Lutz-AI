CREATE TABLE memory (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS whatsapp_identity (
    lid TEXT PRIMARY KEY,
    user_id TEXT,
    phone TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
