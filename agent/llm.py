from agent.providers.factory import LLMFactory


def get_llm(state):

    provider = state.get("provider", "openai")

    if provider == "groq":
        return LLMFactory.groq()

    return LLMFactory.openai()
