import os
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from agent.tools import search_protocol
from agent.guardrails import apply_guardrails
from dotenv import load_dotenv

load_dotenv()

os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=300, temperature=0.5)


class AgentState(TypedDict):
    """Define the state structure for the agent."""
    input: str
    contexto: str
    resposta: str


def node_rag(state):
    """Node responsible for retrieving relevant information based on the user's input."""
    contexto = search_protocol(state["input"])
    state["contexto"] = contexto
    return state


def node_llm(state):
    """Node responsible for generating a response using the LLM based on the retrieved context and user input."""
    messages = [
        SystemMessage(
            content="Você é um agente especializado em saúde da mulher, baseado em diretrizes oficiais."
                    "Você NUNCA deve mencionar nomes de medicamentos específicos (como Paracetamol, Ibuprofeno, etc)."
                    "Se te perguntarem sobre remédios, explique que não pode prescrever e sugira que a usuária procure um médico ou enfermeira para avaliação."
        ),
        HumanMessage(
            content=f"""
    Contexto oficial:
    {state['contexto']}

    Pergunta da paciente:
    {state['input']}
    """
        )
    ]

    resposta = llm.invoke(messages).content
    state["resposta"] = apply_guardrails(resposta)
    return state


def node_guardrails(state):
    """Node responsible for applying guardrails to the generated response, ensuring it adheres to safety and ethical guidelines."""
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
