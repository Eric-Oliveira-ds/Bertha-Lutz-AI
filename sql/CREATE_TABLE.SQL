CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT,
    cpf TEXT UNIQUE,
    date_birth DATE,
    phone TEXT
);

CREATE TABLE memory (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE clinical_profile (
    id SERIAL PRIMARY KEY,
    user_id INT,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE followup_tasks (
    id SERIAL PRIMARY KEY,
    user_id INT,
    status TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE human_review_queue (
    id SERIAL PRIMARY KEY,
    user_id INT,
    message TEXT,
    risk_level TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

