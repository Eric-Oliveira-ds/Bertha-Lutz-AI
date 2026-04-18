import os
import subprocess
import asyncio
import edge_tts
import uuid


async def _generate_tts(text: str, mp3_file: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice="pt-BR-FranciscaNeural"  # 🔥 voz feminina profissional
    )
    await communicate.save(mp3_file)


def text_to_speech(text: str):
    uid = str(uuid.uuid4())
    mp3_file = f"{uid}.mp3"
    ogg_file = f"{uid}.ogg"

    try:
        # gera MP3
        asyncio.run(_generate_tts(text, mp3_file))

        if not os.path.exists(mp3_file):
            print("Erro: MP3 não gerado")
            return None

        # converte para OGG (padrão WhatsApp)
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", mp3_file,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "libopus",
            ogg_file
        ], check=True)

        if not os.path.exists(ogg_file):
            print("Erro: OGG não gerado")
            return None

        return ogg_file

    except Exception as e:
        print("Erro TTS:", e)
        return None

    finally:
        if os.path.exists(mp3_file):
            try:
                os.remove(mp3_file)
            except:
                pass
