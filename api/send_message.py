import requests
import os
from dotenv import load_dotenv
import base64

load_dotenv()

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
INSTANCE_NAME = os.getenv("INSTANCE_NAME")


def send_whatsapp_message(number: str, text: str):
    """Send a WhatsApp message using the Evolution API."""
    clean_number = "".join(filter(str.isdigit, number))
    if len(clean_number) == 11 and clean_number[2] == "9":
        clean_number = clean_number[:2] + clean_number[3:]

    if not clean_number.startswith("55"):
        clean_number = "55" + clean_number

    base_url = EVOLUTION_API_URL.rstrip('/')
    url = f"{base_url}/message/sendText/{INSTANCE_NAME}"

    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "number": clean_number,
        "text": text,
        "delay": 1200,
        "linkPreview": False
    }

    try:
        print(f"Tentando enviar para: {url} | Número: {clean_number}")
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code not in [200, 201]:
            print(
                f"Falha na Evolution API: {response.status_code} - {response.text}")
            return False

        print("Mensagem enviada com sucesso!")
        return True
    except Exception as e:
        print(f"Erro de conexão com a Evolution API: {e}")
        return False


def send_whatsapp_audio(number: str, audio_path: str):
    clean_number = "".join(filter(str.isdigit, number))

    if not clean_number.startswith("55"):
        clean_number = "55" + clean_number

    url = f"{EVOLUTION_API_URL.rstrip('/')}/message/sendMedia/{INSTANCE_NAME}"

    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        with open(audio_path, "rb") as audio:
            audio_base64 = base64.b64encode(audio.read()).decode("utf-8")

        payload = {
            "number": clean_number,
            "mediatype": "audio",
            "mimetype": "audio/ogg; codecs=opus",  # 🔥 CRÍTICO
            "caption": "",
            "media": audio_base64,
            "ptt": True  # 🔥 vira áudio de voz (bolinha)
        }

        print(f"Enviando áudio para: {clean_number}")

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code not in [200, 201]:
            print("Erro ao enviar áudio:", response.text)
            return False

        print("Áudio enviado com sucesso!")
        return True

    except Exception as e:
        print("Erro envio áudio:", e)
        return False
