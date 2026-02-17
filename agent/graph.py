import os
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import AIMessage
from typing import TypedDict
from agent.tools import search_protocol
from agent.guardrails import apply_guardrails
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=300, temperature=0.2)


class AgentState(TypedDict):
    """Define the state structure for the agent."""
    input: str
    history: list
    contexto: str
    resposta: str


def node_rag(state):
    """Node responsible for retrieving relevant information based on the user's input."""
    contexto = search_protocol(state["input"])
    state["contexto"] = contexto
    return state


def node_llm(state):
    messages = [
        SystemMessage(
            content="Você é um agente especializado em saúde da mulher, baseado em diretrizes oficiais."
                    "Você NUNCA deve mencionar nomes de medicamentos específicos (como Paracetamol, Ibuprofeno, etc)."
                    "Se te perguntarem sobre remédios, explique que não pode prescrever e sugira que a usuária procure um médico ou enfermeira para avaliação."
        )
    ]

    # adiciona histórico
    for msg in state["history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # adiciona pergunta atual com contexto
    messages.append(
        HumanMessage(
            content=f"""
    Contexto oficial:
    {state['contexto']}

    Pergunta da paciente:
    {state['input']}
    """
        )
    )

    resposta = llm.invoke(messages).content
    state["resposta"] = resposta
    return state


def node_guardrails(state):
    """Node responsible for applying guardrails to the LLM's response."""
    state["resposta"] = apply_guardrails(state["resposta"])
    return state


def agent_graph():
    """Function to create and compile the agent's state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("rag", node_rag)
    graph.add_node("llm", node_llm)
    graph.add_node("guardrails", node_guardrails)

    graph.set_entry_point("rag")

    graph.add_edge("rag", "llm")
    graph.add_edge("llm", "guardrails")

    app = graph.compile()

    return app
