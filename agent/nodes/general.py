from time import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent.tools.output_parser import clean_tts_text
from agent.metrics.metrics import llm_latency_seconds, llm_tokens_total

llm_general = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=150
)

def general_node(state):
    start = time()

    messages = [
        SystemMessage(content="""
Você é uma assistente de saúde amigável, chamada Bertha Lutz.
Responda de forma curta, acolhedora e educada.
Nunca dê diagnósticos, prescrições ou recomendações médicas.
Se o usuário cumprimentar, cumprimente de volta.
Se não souber algo, sugira que o usuário fale com um profissional de saúde.
"""),
        HumanMessage(content=state["input"])
    ]

    response = llm_general.invoke(messages)

    duration = time() - start
    llm_latency_seconds.observe(duration)

    usage = response.response_metadata.get("token_usage", {})
    if "prompt_tokens" in usage:
        llm_tokens_total.labels(type="prompt").inc(usage["prompt_tokens"])
    if "completion_tokens" in usage:
        llm_tokens_total.labels(type="completion").inc(usage["completion_tokens"])

    state["resposta"] = clean_tts_text(response.content)
    return state
