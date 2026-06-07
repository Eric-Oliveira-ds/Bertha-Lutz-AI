import json
from time import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from agent.tools.output_parser import clean_tts_text
from agent.services.clinical_db import (
    save_clinical_profile
)

from agent.metrics.metrics import (
    llm_latency_seconds,
    llm_tokens_total
)

llm_collector = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.2,
    max_tokens=400
)


def collector_node(state):

    start = time()

    messages = [

        SystemMessage(
            content="""
Você é um agente de coleta clínica.

Objetivos:
- coletar informações estruturadas
- responder de forma curta
- não prescrever medicamentos

Extraia:
- idade
- gravidez
- última menstruação
- preventivo
- mamografia
- sintomas

Retorne JSON válido.

Formato:

{
  "response": "...",
  "structured_data": {
      "pregnant": false,
      "last_pap_smear": "2025-10-01"
  },
  "followup_needed": true
}
"""
        ),

        HumanMessage(
            content=f"""
Contexto:
{state["contexto"]}

Paciente:
{state["input"]}
"""
        )
    ]

    response = llm_collector.invoke(messages)

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

    state["resposta"] = clean_tts_text(
        data["response"]
    )

    state["structured_data"] = data["structured_data"]

    state["followup_needed"] = data["followup_needed"]

    # salva no banco
    save_clinical_profile(
        user_id=state["user_id"],
        data=state["structured_data"]
    )

    return state
