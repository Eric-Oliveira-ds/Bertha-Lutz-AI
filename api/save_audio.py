import base64
import uuid


def save_base64_audio(base64_data: str):

    filename = f"{uuid.uuid4()}.ogg"

    audio_bytes = base64.b64decode(base64_data)

    with open(filename, "wb") as f:
        f.write(audio_bytes)

    return filename
