from agent.tools.tools import search_protocol


def rag_node(state):

    contexto = search_protocol(state["input"])

    state["contexto"] = contexto

    return state
