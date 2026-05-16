import requests
import uuid


def download_whatsapp_audio(url):

    filename = f"{uuid.uuid4()}.ogg"

    response = requests.get(url)

    if response.status_code != 200:
        return None

    with open(filename, "wb") as f:
        f.write(response.content)

    return filename
