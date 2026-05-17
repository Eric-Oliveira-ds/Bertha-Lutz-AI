from prometheus_client import Counter, Histogram
from prometheus_client import Gauge

# Total de tokens consumidos
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total de tokens consumidos pelo LLM",
    ["type"]  # prompt / completion
)

# Latência do LLM
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "Tempo de resposta do LLM"
)

# Fallback / bloqueios do guardrail
guardrail_blocks_total = Counter(
    "guardrail_blocks_total",
    "Total de respostas bloqueadas por guardrails"
)

# Métricas de avaliação (DeepEval)
rag_faithfulness_score = Gauge(
    "rag_faithfulness_score",
    "Faithfulness score from DeepEval"
)

# Métricas de relevância (DeepEval)
rag_relevancy_score = Gauge(
    "rag_relevancy_score",
    "Answer relevancy score from DeepEval"
)
