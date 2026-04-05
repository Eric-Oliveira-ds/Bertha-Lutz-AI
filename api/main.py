from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from agent.graph import agent_graph
from agent.memory import save_memory, load_memory, SessionLocal
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from time import time
from datetime import datetime
from api.send_message import send_whatsapp_message

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
                f"Olá {name}! Sou a Bertha, sua assistente de saúde. 🌸\n\n"
                "Para começarmos, qual foi a última vez que você foi a uma UBS "
                "ou hospital para realizar exames de rotina?"
            )

            sucesso_whatsapp = send_whatsapp_message(phone, primeira_pergunta)

            # Opcional: Salvar essa primeira interação na memória do agente
            if sucesso_whatsapp:
                save_memory(str(user_id), "assistant", primeira_pergunta)

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


@app.post("/chat")
def chat(user_id: str = Form(...), message: str = Form(...)):

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

    print(f"LLM latency: {duration:.3f}s")

    return {"response": response}
