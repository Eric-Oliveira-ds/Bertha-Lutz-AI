import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi import BackgroundTasks
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from agent.graph import agent_graph
from agent.memory import save_memory, load_memory, SessionLocal
from prometheus_fastapi_instrumentator import Instrumentator
from time import time
from datetime import datetime
from api.send_message import send_whatsapp_message
from api.stt import transcribe_audio as speech_to_text
from api.tts import text_to_speech
from api.send_message import send_whatsapp_audio
from api.validar_cpf import validar_cpf
from api.db_create_tables import create_tables


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend/templates")

agent = agent_graph()

Instrumentator().instrument(app).expose(app)

# ---------- DB INIT ----------
create_tables()


# ---------- ROUTES ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/register")
def register(name: str = Form(...), cpf: str = Form(...), date_birth: str = Form(...), phone: str = Form(...)):

    validar_cpf(cpf)
    # Converte a data (ajuste o formato se necessário para o seu front-end)
    date_obj = datetime.strptime(date_birth, "%d/%m/%Y").date()

    with SessionLocal() as session:
        try:
            # Inserção no banco de dados
            result = session.execute(
                text("""
                    INSERT INTO users (name, cpf, date_birth, phone)
                    VALUES (:name, :cpf, :date_birth, :phone)
                    RETURNING id
                """),
                {"name": name, "cpf": cpf, "date_birth": date_obj, "phone": phone}
            )

            user_id = result.fetchone()[0]
            session.commit()

            # --- INTEGRAÇÃO WHATSAPP ---
            # Mensagem estratégica de boas-vindas
            primeira_pergunta = (
                f"Olá {name.split()[0]}! Sou a sua assistente de saúde, seja bem-vindo(a)!"
            )
            try:
                audio_file = text_to_speech(primeira_pergunta)
            except Exception as e:
                print(f"[TTS] Erro ao gerar áudio: {e}")
                audio_file = None

            sucesso_whatsapp = False

            if audio_file and os.path.exists(audio_file):
                sucesso_whatsapp = send_whatsapp_audio(phone, audio_file)
            else:
                sucesso_whatsapp = send_whatsapp_message(phone, primeira_pergunta)

            if sucesso_whatsapp:
                save_memory(str(user_id), "assistant", primeira_pergunta)

            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)

            return {
                "user_id": user_id,
                "whatsapp_sent": sucesso_whatsapp
            }

        except IntegrityError:
            session.rollback()
            raise HTTPException(
                status_code=400,
                detail="CPF já cadastrado"
            )


@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request,background_tasks: BackgroundTasks):

    try:
        body = await request.json()

        print("\n\n===== WEBHOOK RECEBIDO =====")
        print(body)
        print("============================\n\n")

        event = body.get("event", "").lower()

        if event != "messages.upsert":
            return {"status": "ignored"}

        data = body.get("data", {})
        key = data.get("key", {})

        # ignora mensagens enviadas pelo próprio bot
        if key.get("fromMe"):
            return {"status": "ignored"}

        # ------------------------------------------------
        # PEGA DIRETAMENTE O remoteJidAlt
        # Ex:
        # 55759xxxxxxxx@s.whatsapp.net
        # ------------------------------------------------
        remote_jid_alt = key.get("remoteJidAlt")

        if not remote_jid_alt:
            return {"status": "remote_jid_alt_not_found"}

        # extrai somente o telefone
        phone = remote_jid_alt.split("@")[0]

        if not phone:
            return {"status": "phone_not_found"}

        # ------------------------------------------------
        # BUSCA USER PELO TELEFONE
        # ------------------------------------------------
        with SessionLocal() as session:

            result = session.execute(
                text("""
                    SELECT id, phone
                    FROM users
                    WHERE phone LIKE :suffix
                    LIMIT 1
                """),
                {"suffix": f"%{phone[-8:]}"}
            ).fetchone()

            if not result:
                return {"status": "user_not_found"}

            user_id = str(result[0])
            phone = normalize_phone(result[1])

        # ------------------------------------------------
        # PARSE DA MENSAGEM
        # ------------------------------------------------
        message_data = data.get("message", {})
        print(message_data.keys())
        message_text = None

        # TEXTO NORMAL
        if "conversation" in message_data:
            message_text = message_data["conversation"]

        # TEXTO EXTENDIDO
        elif "extendedTextMessage" in message_data:
            message_text = message_data["extendedTextMessage"]["text"]

        # ÁUDIO
        elif "audioMessage" in message_data:

            base64_audio = message_data.get("base64")

            if not base64_audio:
                return {"status": "audio_base64_not_found"}

            transcribed_text = speech_to_text(base64_audio)

            print("TRANSCRIÇÃO:", transcribed_text)

            if not transcribed_text:
                return {"status": "audio_transcription_error"}

            message_text = transcribed_text.strip()

            print(f"[STT] Texto transcrito: {message_text}")

        if not message_text:
            return {"status": "empty_message"}

        print(f"[WEBHOOK] message={message_text}")
        print(f"[WEBHOOK] phone={phone}")
        print(f"[WEBHOOK] user_id={user_id}")

        # ------------------------------------------------
        # PROCESSAMENTO ASSÍNCRONO
        # ------------------------------------------------
        background_tasks.add_task(
            process_message,
            user_id,
            phone,
            message_text
        )

        return {"status": "processing"}

    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")
        return {"status": "parse_error"}


def process_message(user_id: str, phone: str, message: str):

    try:
        history = load_memory(user_id)

        start = time()

        result = agent.invoke({
            "input": message,
            "history": history
        })

        duration = time() - start

        # ⚠️ fallback defensivo
        response = result.get("resposta") or result.get("output") or "Desculpa, não consegui entender. Pode repetir?"

        # salvar memória
        save_memory(user_id, "user", message)
        save_memory(user_id, "assistant", response)

        print(f"[WEBHOOK] user={user_id} phone={phone} latency={duration:.2f}s")

        # 🎤 TTS
        try:
            audio_file = text_to_speech(response)
        except Exception as e:
            print(f"[TTS] Erro ao gerar áudio: {e}")
            audio_file = None

        # 📤 envio
        if audio_file and os.path.exists(audio_file):
            send_whatsapp_audio(phone, audio_file)
            os.remove(audio_file)
        else:
            send_whatsapp_message(phone, response)

    except Exception as e:
        print(f"[PROCESS_MESSAGE] Erro geral: {e}")


def normalize_phone(phone: str):
    return "".join(filter(str.isdigit, phone))
