import base64
import uuid
import subprocess
from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


def transcribe_audio(base64_audio: str):

    audio_id = str(uuid.uuid4())

    ogg_path = f"/tmp/{audio_id}.ogg"
    wav_path = f"/tmp/{audio_id}.wav"

    # salva ogg
    with open(ogg_path, "wb") as f:
        f.write(base64.b64decode(base64_audio))

    # converte para wav
    subprocess.run([
        "ffmpeg",
        "-i", ogg_path,
        "-ar", "16000",
        "-ac", "1",
        wav_path,
        "-y"
    ], check=True)

    # whisper
    segments, info = model.transcribe(
        wav_path,
        language="pt"
    )

    text = " ".join([segment.text for segment in segments])

    return text.strip()
