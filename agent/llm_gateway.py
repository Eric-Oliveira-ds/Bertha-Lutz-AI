from providers.factory import LLMFactory


class LLMGateway:

    @staticmethod
    def _get_provider(provider):

        return (
            LLMFactory.groq()
            if provider == "groq"
            else LLMFactory.openai()
        )

    @staticmethod
    def _get_fallback(provider):

        return (
            "openai"
            if provider == "groq"
            else "groq"
        )

    @staticmethod
    def invoke(messages, provider="openai"):

        try:

            return (
                LLMGateway
                ._get_provider(provider)
                .invoke(messages)
            )

        except Exception as e:

            error_text = str(e).lower()

            recoverable = any(
                x in error_text
                for x in [
                    "429",
                    "rate limit",
                    "timeout",
                    "connection",
                    "503",
                    "502",
                ]
            )

            if not recoverable:
                raise

            fallback_provider = (
                LLMGateway
                ._get_fallback(provider)
            )

            print(
                f"Fallback: "
                f"{provider} -> {fallback_provider}"
            )

            return (
                LLMGateway
                ._get_provider(fallback_provider)
                .invoke(messages)
            )
