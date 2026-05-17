from sqlalchemy import text

from agent.memory.memory import SessionLocal


def human_review_node(state):

    if not state["human_review"]:
        return state

    with SessionLocal() as session:

        session.execute(
            text("""
INSERT INTO human_review_queue (
    user_id,
    message,
    risk_level,
    resolved
)
VALUES (
    :user_id,
    :message,
    :risk_level,
    false
)
"""),
            {
                "user_id": state["user_id"],
                "message": state["input"],
                "risk_level": state["risk_level"]
            }
        )

        session.commit()

    return state
