import re


def clean_tts_text(text: str) -> str:
    """
    Remove markdown e caracteres que prejudicam TTS.
    """

    # remove markdown bold/italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # remove headers markdown
    text = re.sub(r"#+\s", "", text)

    # remove bullets markdown
    text = re.sub(r"[-•]\s", "", text)

    # remove links markdown
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)

    # remove múltiplos espaços
    text = re.sub(r"\s+", " ", text)

    return text.strip()