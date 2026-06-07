from agent.memory.memory import SessionLocal
from sqlalchemy import JSON, Column, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ClinicalProfile(Base):
    __tablename__ = 'clinical_profile'

    id = Column(Integer, primary_key=True)

    # corrigido
    user_id = Column(Integer)

    data = Column(JSON)


def save_clinical_profile(user_id: int, data: dict):

    profile = ClinicalProfile(
        user_id=int(user_id),
        data=data
    )

    with SessionLocal() as session:
        session.add(profile)
        session.commit()
