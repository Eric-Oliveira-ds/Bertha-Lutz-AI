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


# ---------- DB INIT ----------

def create_tables():
    with SessionLocal() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT,
                cpf TEXT UNIQUE
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
def register(name: str = Form(...), cpf: str = Form(...)):
    with SessionLocal() as session:
        try:

            result = session.execute(
                text("""
                    INSERT INTO users (name, cpf)
                    VALUES (:name, :cpf)
                    RETURNING id
                """),
                {"name": name, "cpf": cpf}
            )

            user_id = result.fetchone()[0]
            session.commit()

            return {"user_id": user_id}

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
