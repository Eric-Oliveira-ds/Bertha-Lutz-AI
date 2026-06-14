def choose_provider(state):

    route = state.get("route")
    risk = state.get("risk_level")

    # Casos críticos
    if risk == "alto":
        return "openai"

    # Educação em saúde
    if route == "geral":
        return "groq"

    # Followup
    if route == "followup":
        return "groq"

    # Coleta clínica
    if route == "coleta":
        return "openai"

    return "openai"
