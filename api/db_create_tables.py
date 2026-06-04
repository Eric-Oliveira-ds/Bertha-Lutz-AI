from agent.memory.memory import SessionLocal
from sqlalchemy import text


def create_tables():
    with SessionLocal() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT,
                cpf TEXT UNIQUE,
                date_birth DATE,
                phone TEXT
            );
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS memory (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS followup_tasks (
                id SERIAL PRIMARY KEY,
                user_id INT,
                message TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS clinical_profile (
                id SERIAL PRIMARY KEY,
                user_id INT,
                data JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS human_review_queue (
                id SERIAL PRIMARY KEY,
                user_id INT,
                message TEXT,
                risk_level TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))

        session.commit()
