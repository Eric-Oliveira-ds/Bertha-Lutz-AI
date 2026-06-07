from time import time
from pydantic import BaseModel
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent.services.risk_engine import calculate_risk
from agent.metrics.metrics import llm_latency_seconds, llm_tokens_total


class SupervisorOutput(BaseModel):
    route: Literal[
        "coleta",
        "geral",
        "followup",
        "humano"
    ]

    risk_level: Literal[
        "baixo",
        "moderado",
        "alto"
    ]

    confidence: float
    human_review: bool


llm_router = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    max_tokens=80
).with_structured_output(
    SupervisorOutput,
    include_raw=True
)


def supervisor_node(state):
    start = time()

    messages = [
        SystemMessage(content="""
        Você é um supervisor clínico.

        Analise:
        - intenção
        - risco real de saúde
        - necessidade humana

        Retorne APENAS JSON válido.

        Rotas possíveis:
        - coleta:
                usar SOMENTE quando o usuário estiver:
                - descrevendo sintomas próprios
                - informando idade
                - gravidez
                - exames pessoais
                - histórico clínico próprio
                - respondendo perguntas clínicas

                Exemplos:
                "estou com febre"
                "tenho diabetes"
                "minha menstruação atrasou"

        - geral:
                usar para:
                - perguntas informativas
                - educação em saúde
                - exames preventivos
                - diretrizes do SUS
                - prevenção
                - conversas gerais
                - saudações

                Exemplos:
                "quais exames mulheres devem fazer?"
                "o que é hipertensão?"
                "como funciona o preventivo?"

        - humano (apenas para risco alto confirmado)
        - followup (quando é um acompanhamento agendado)

        Níveis de risco:
        - baixo (saudações, conversas informais, informações gerais)
        - moderado (sintomas leves sem urgência)
        - alto (sintomas graves, emergências, violência, ideação suicida)

        ⚠️ IMPORTANTE: Saudações como "Bom dia", "Boa noite", "Olá" são risco BAIXO e rota GERAL, NUNCA acione human_review.

        Exemplo de resposta para saudação:
        {"route": "geral", "risk_level": "baixo", "confidence": 0.99, "human_review": false}
        """),

        HumanMessage(content=f"""
        Contexto:
        {state["contexto"]}

        Mensagem:
        {state["input"]}
        """)
        ]

    result = llm_router.invoke(messages)
    raw = result["raw"]
    parsed = result["parsed"]
    data = parsed.model_dump()

    # Métricas
    duration = time() - start
    llm_latency_seconds.observe(duration)
    usage = raw.response_metadata.get("token_usage", {})

    if "prompt_tokens" in usage:
        llm_tokens_total.labels(type="prompt").inc(usage["prompt_tokens"])
    if "completion_tokens" in usage:
        llm_tokens_total.labels(type="completion").inc(usage["completion_tokens"])

    # Risk engine (prioridade máxima)
    risk_data = calculate_risk(state["input"])
    if risk_data["risk_level"] == "alto":
        state["risk_level"] = "alto"
        state["human_review"] = True
    else:
        state["risk_level"] = risk_data["risk_level"]
        state["human_review"] = risk_data["human_review"]

    state["route"] = data.get("route")
    state["confidence"] = data.get("confidence")

    return state
