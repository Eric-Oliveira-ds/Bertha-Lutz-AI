from time import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent.tools.output_parser import clean_tts_text
from agent.metrics.metrics import llm_latency_seconds, llm_tokens_total


llm_risk = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def risk_node(state):

    start = time()

    if state["risk_level"] != "alto":
        return state

    messages = [

        SystemMessage(
            content="""
Você é um agente clínico de risco.

Sua função:
- orientar busca de atendimento
- evitar diagnóstico
- evitar prescrição
- acionar revisão humana
"""
        ),

        HumanMessage(
            content=state["input"]
        )
    ]

    response = llm_risk.invoke(messages)

    duration = time() - start

    llm_latency_seconds.observe(duration)

    usage = response.response_metadata.get("token_usage", {})

    if "prompt_tokens" in usage:
        llm_tokens_total.labels(type="prompt").inc(
            usage["prompt_tokens"]
        )

    if "completion_tokens" in usage:
        llm_tokens_total.labels(type="completion").inc(
            usage["completion_tokens"]
        )

    state["resposta"] = clean_tts_text(
        response.content
    )

    state["human_review"] = True

    return state
