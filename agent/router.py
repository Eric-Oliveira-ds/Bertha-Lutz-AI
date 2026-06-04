def route_decision(state):
    if state["human_review"]:
        return "human_review"
    if state["risk_level"] == "alto":
        return "risk"

    route = state.get("route", "geral")

    if route == "coleta":
        return "collector"
    if route == "followup":
        return "followup"
    if route == "geral":
        return "general"
    # fallback
    return "general"
