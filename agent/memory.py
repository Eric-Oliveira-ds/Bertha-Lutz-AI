from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/postgres",
    pool_pre_ping=True
)

SessionLocal = sessionmaker(bind=engine)


def save_memory(user_id: str, role: str, content: str):
    """
    Saves a memory entry to the database.
    Args:
    ----
        user_id (str): The ID of the user associated with the memory.
        role (str): The role of the memory (e.g., "user", "assistant").
        content (str): The content of the memory entry.

    """
    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO memory (user_id, role, content)
                VALUES (:user_id, :role, :content)
            """),
            {
                "user_id": user_id,
                "role": role,
                "content": content
            }
        )
        session.commit()


def load_memory(user_id: str):
    """
    Loads memory entries for a given user from the database.

    Args:
    ----
        user_id (str): The ID of the user whose memory entries are to be loaded.

    Returns:
    -------
        List of tuples containing role and content of memory entries for the specified user.

    """
    with SessionLocal() as session:
        result = session.execute(
            text("""
                SELECT role, content
                FROM memory
                WHERE user_id = :user_id
                ORDER BY created_at
            """),
            {"user_id": user_id}
        )
        return result.fetchall()
