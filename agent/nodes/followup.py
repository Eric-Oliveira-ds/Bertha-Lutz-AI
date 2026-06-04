from sqlalchemy import text

from agent.memory.memory import SessionLocal


def followup_node(state):
    if not state["followup_needed"]:
        return state

    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO followup_tasks (user_id, status)
                VALUES (:user_id, 'pending')
            """),
            {"user_id": state["user_id"]}
        )
        session.commit()

    # 🔥 Resposta para o usuário
    state["resposta"] = (
        "✅ Entendi! Vou acompanhar esse assunto e entrarei em contato novamente em breve. "
        "Se precisar falar antes, é só me chamar."
    )
    return state
