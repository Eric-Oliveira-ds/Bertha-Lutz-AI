from agent.guardrails import apply_guardrails


def guardrails_node(state):
    # Se não houver resposta, usa uma padrão
    if "resposta" not in state or not state["resposta"]:
        state["resposta"] = (
            "Desculpe, tive um problema ao gerar minha resposta. "
            "Por favor, tente de novo ou entre em contato com a equipe de suporte."
        )
    state["resposta"] = apply_guardrails(state["resposta"])
    return state
