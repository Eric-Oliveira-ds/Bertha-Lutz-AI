HIGH_RISK_TERMS = [

    "sangramento intenso",
    "dor no peito",
    "desmaio",
    "falta de ar",
    "tentando me matar",
    "violência",
    "abuso",
    "convulsão",
    "grávida sangrando",
    "caroço crescendo"
]


def calculate_risk(message: str):

    message = message.lower()

    score = 0

    matched_terms = []

    for term in HIGH_RISK_TERMS:

        if term in message:

            score += 1
            matched_terms.append(term)

    if score >= 2:

        return {
            "risk_level": "alto",
            "human_review": True,
            "matched_terms": matched_terms
        }

    if score == 1:

        return {
            "risk_level": "moderado",
            "human_review": False,
            "matched_terms": matched_terms
        }

    return {
        "risk_level": "baixo",
        "human_review": False,
        "matched_terms": []
    }
