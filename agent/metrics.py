from prometheus_client import Counter, Histogram

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
