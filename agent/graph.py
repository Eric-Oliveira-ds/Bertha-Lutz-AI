from langgraph.graph import StateGraph
from langgraph.graph import END

from agent.state import AgentState

from agent.router import route_decision

from agent.nodes.rag import rag_node
from agent.nodes.supervisor import supervisor_node
from agent.nodes.collector import collector_node
from agent.nodes.risk import risk_node
from agent.nodes.followup import followup_node
from agent.nodes.human_review import human_review_node
from agent.nodes.guardrails import guardrails_node
from agent.nodes.general import general_node


def agent_graph():

    graph = StateGraph(AgentState)

    graph.add_node("rag", rag_node)

    graph.add_node(
        "supervisor",
        supervisor_node
    )

    graph.add_node(
        "collector",
        collector_node
    )

    graph.add_node(
        "risk",
        risk_node
    )

    graph.add_node(
        "followup",
        followup_node
    )

    graph.add_node(
        "human_review",
        human_review_node
    )

    graph.add_node(
        "guardrails",
        guardrails_node
    )

    graph.add_node(
        "general", 
        general_node)

    graph.set_entry_point("rag")

    graph.add_edge(
        "rag",
        "supervisor"
    )

    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "collector": "collector",
            "risk": "risk",
            "followup": "followup",
            "human_review": "human_review",
            "general": "general",
            "guardrails": "guardrails"
        }
    )

    graph.add_edge(
        "collector",
        "guardrails"
    )

    graph.add_edge(
        "risk",
        "human_review"
    )

    graph.add_edge(
        "human_review",
        "guardrails"
    )

    graph.add_edge(
        "followup",
        "guardrails"
    )

    graph.add_edge(
        "guardrails",
        END
    )

    graph.add_edge(
        "general",
        "guardrails")

    app = graph.compile()

    return app
