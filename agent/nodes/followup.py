from sqlalchemy import text

from agent.memory.memory import SessionLocal


def followup_node(state):

    if not state["followup_needed"]:
        return state

    with SessionLocal() as session:

        session.execute(
            text("""
INSERT INTO followup_tasks (
    user_id,
    status
)
VALUES (
    :user_id,
    'pending'
)
"""),
            {
                "user_id": state["user_id"]
            }
        )

        session.commit()

    return state