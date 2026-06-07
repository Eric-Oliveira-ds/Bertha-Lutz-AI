import json
import re
from time import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent.services.risk_engine import calculate_risk
from agent.metrics.metrics import llm_latency_seconds, llm_tokens_total

llm_router = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    max_tokens=200
)


def extract_json_from_text(text: str) -> dict | None:
    """Tenta extrair o primeiro JSON válido de uma string."""
    # Procura por {...} ou [...] (nosso caso é objeto)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Se não achar, tenta parsear direto (caso seja só o JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


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
        - coleta (quando o paciente fornece dados clínicos)
        - followup (quando é um acompanhamento agendado)
        - geral (para conversas casuais, saudações, despedidas)
        - humano (apenas para risco alto confirmado)

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

    response = llm_router.invoke(messages)
    content = response.content.strip()

    # Log para depuração (opcional, mas ajuda)
    print(f"[SUPERVISOR] Raw response: {content[:200]}")

    # Fallback padrão
    default_data = {
        "route": "geral",
        "risk_level": "baixo",
        "confidence": 0.5,
        "human_review": False
    }

    if not content:
        print("[SUPERVISOR] Resposta vazia do LLM, usando fallback")
        data = default_data
    else:
        data = extract_json_from_text(content)
        if data is None:
            print(f"[SUPERVISOR] JSON inválido recebido: {content[:200]}")
            data = default_data

    # Métricas
    duration = time() - start
    llm_latency_seconds.observe(duration)

    usage = response.response_metadata.get("token_usage", {})
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

    state["route"] = data.get("route", default_data["route"])
    state["confidence"] = data.get("confidence", default_data["confidence"])

    return state
