from agent.guardrails import apply_guardrails


def guardrails_node(state):

    state["resposta"] = apply_guardrails(
        state["resposta"]
    )

    return state
