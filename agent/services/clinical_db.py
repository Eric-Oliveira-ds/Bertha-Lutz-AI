from agent.memory.memory import SessionLocal
from sqlalchemy import JSON, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ClinicalProfile(Base):
    __tablename__ = 'clinical_profile'
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    data = Column(JSON)


def save_clinical_profile(user_id: str, data: dict):
    profile = ClinicalProfile(user_id=user_id, data=data)
    with SessionLocal() as session:
        session.add(profile)
        session.commit()
