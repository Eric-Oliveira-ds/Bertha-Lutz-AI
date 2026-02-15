import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


def apply_guardrails(resposta: str) -> str:
    # 1. Prompt mais diretivo e severo
    system_prompt = (
        "Você é um auditor de segurança médica rigoroso. "
        "Sua única tarefa é identificar se o assistente sugeriu NOMES DE MEDICAMENTOS específicos "
        "(ex: paracetamol, ibuprofeno, buscopan) ou deu um diagnóstico. "
        "Se o assistente citou qualquer nome de remédio, você DEVE responder 'INSEGURO'. "
        "Se o assistente apenas deu conselhos gerais e mandou procurar um médico, responda 'SEGURO'. "
        "Responda APENAS com a palavra 'SEGURO' ou 'INSEGURO'."
    )

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analise se esta resposta contém nomes de medicamentos: {resposta}"}
        ],
        temperature=0,
        max_tokens=5
    )

    veredito = chat_completion.choices[0].message.content.strip().upper()

    # Se a resposta contiver 'INSEGURO' em qualquer parte, bloqueamos.
    if "INSEGURO" in veredito:
        return (
            "⚠️ **Aviso de Segurança Bertha Lutz:**\n"
            "Não posso recomendar medicamentos, dosagens ou fornecer diagnósticos. "
            "Embora existam opções comuns para dor, cada organismo é único e a automedicação é arriscada.\n\n"
            "👉 **Recomendação:** Procure a Unidade Básica de Saúde (UBS) mais próxima ou um ginecologista.\n"
            "---"
        )

    return resposta
