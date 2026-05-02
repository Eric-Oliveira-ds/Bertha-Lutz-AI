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
from pydantic import BaseModel
from time import time
from datetime import datetime
from api.send_message import send_whatsapp_message
from api.tts import text_to_speech
from api.send_message import send_whatsapp_audio
import os

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


class RegisterRequest(BaseModel):
    name: str
    cpf: str


def validar_cpf(cpf: str):
    cpf_limpo = "".join(filter(str.isdigit, cpf))

    if len(cpf_limpo) != 11:
        raise HTTPException(
            status_code=400,
            detail="CPF deve conter 11 dígitos"
        )


# ---------- DB INIT ----------

def create_tables():
    with SessionLocal() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT,
                cpf TEXT UNIQUE,
                date_birth DATE,
                phone TEXT
            );
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS memory (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))

        session.commit()


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
            # Mensagem estratégica de boas-vindas e triagem
            primeira_pergunta = (
                f"Olá {name.split()[0]}! Sou a sua assistente de saúde. Para começarmos, qual foi a última vez que você foi a uma unidade de saúde para realizar exames de rotina?"
            )
            try:
                audio_file = text_to_speech(primeira_pergunta)
            except Exception as e:
                print(f"Erro ao gerar áudio: {e}")
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


class WhatsAppMessage(BaseModel):
    phone: str
    message: str


@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request, background_tasks: BackgroundTasks):

    try:
        body = await request.json()
    except Exception:
        print("Webhook sem JSON válido")
        return {"status": "invalid_json"}

    try:
        if body["data"]["key"]["fromMe"]:
            return {"status": "ignored"}

        phone = body["data"]["key"]["remoteJid"].split("@")[0]

        msg_data = body["data"]["message"]

        if "conversation" in msg_data:
            message = msg_data["conversation"]
        elif "extendedTextMessage" in msg_data:
            message = msg_data["extendedTextMessage"]["text"]
        else:
            return {"status": "ignored"}

        phone = normalize_phone(phone)

    except Exception as e:
        print("Erro ao parsear webhook:", body, e)
        return {"status": "parse_error"}


def process_message(user_id: str, phone: str, message: str):

    history = load_memory(user_id)

    start = time()

    result = agent.invoke({
        "input": message,
        "history": history
    })

    duration = time() - start

    response = result["resposta"]

    save_memory(user_id, "user", message)
    save_memory(user_id, "assistant", response)

    print(f"[WEBHOOK] user={user_id} phone={phone} latency={duration:.2f}s")

    try:
        audio_file = text_to_speech(response)
    except Exception as e:
        print(f"Erro ao gerar áudio: {e}")
        audio_file = None

    if audio_file and os.path.exists(audio_file):
        send_whatsapp_audio(phone, audio_file)
        os.remove(audio_file)
    else:
        send_whatsapp_message(phone, response)

def normalize_phone(number: str):
    clean = "".join(filter(str.isdigit, number))

    if not clean.startswith("55"):
        clean = "55" + clean

    return clean
