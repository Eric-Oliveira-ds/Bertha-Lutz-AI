from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


class LLMFactory:

    @staticmethod
    def openai():
        return ChatOpenAI(
            model="gpt-5.4-mini",
            temperature=0
        )

    @staticmethod
    def groq():
        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )
