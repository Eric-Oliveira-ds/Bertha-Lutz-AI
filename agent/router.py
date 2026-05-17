def route_decision(state):

    if state["human_review"]:
        return "human_review"

    if state["risk_level"] == "alto":
        return "risk"

    route = state["route"]

    if route == "coleta":
        return "collector"

    if route == "followup":
        return "followup"

    return "guardrails"
