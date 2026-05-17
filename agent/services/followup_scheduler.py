from sqlalchemy import text
from agent.memory.memory import SessionLocal
from api.send_message import send_whatsapp_message


def run_followups():

    with SessionLocal() as session:

        patients = session.execute(
            text("""
SELECT
    u.phone,
    cp.data
FROM clinical_profile cp
JOIN users u
ON u.id = cp.user_id
""")
        ).fetchall()

        for patient in patients:

            phone = patient[0]

            message = (
                "Olá. Passando para lembrar "
                "sobre seus exames preventivos."
            )

            send_whatsapp_message(
                phone,
                message
            )
