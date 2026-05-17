from sqlalchemy import text
from agent.memory.memory import SessionLocal


def save_clinical_profile(user_id, data):

    with SessionLocal() as session:

        session.execute(
            text("""
INSERT INTO clinical_profile (
    user_id,
    data
)
VALUES (
    :user_id,
    :data
)
"""),
            {
                "user_id": user_id,
                "data": str(data)
            }
        )

        session.commit()
