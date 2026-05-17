from typing import TypedDict


class AgentState(TypedDict):

    input: str
    history: list

    user_id: str

    contexto: str

    route: str

    risk_level: str
    confidence: float

    structured_data: dict

    followup_needed: bool
    human_review: bool

    resposta: str
