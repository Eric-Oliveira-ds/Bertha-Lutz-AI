import json
from time import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from agent.services.risk_engine import calculate_risk

from agent.metrics.metrics import (
    llm_latency_seconds,
    llm_tokens_total
)

llm_router = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200
)


def supervisor_node(state):

    start = time()

    messages = [

        SystemMessage(
            content="""
Você é um supervisor clínico.

Analise:
- intenção
- risco
- necessidade humana

Retorne JSON válido.

Rotas possíveis:
- coleta
- followup
- geral
- humano

Níveis de risco:
- baixo
- moderado
- alto

Exemplo:

{
  "route": "coleta",
  "risk_level": "baixo",
  "confidence": 0.91,
  "human_review": false
}
"""
        ),

        HumanMessage(
            content=f"""
Contexto:
{state["contexto"]}

Mensagem:
{state["input"]}
"""
        )
    ]

    response = llm_router.invoke(messages)

    data = json.loads(response.content)

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

    risk_data = calculate_risk(state["input"])
    # prioridade máxima
    if risk_data["risk_level"] == "alto":
        state["risk_level"] = "alto"
        state["human_review"] = True
    else:
        state["risk_level"] = risk_data["risk_level"]
        state["human_review"] = risk_data["human_review"]

    state["route"] = data["route"]
    state["confidence"] = data["confidence"]

    return state
