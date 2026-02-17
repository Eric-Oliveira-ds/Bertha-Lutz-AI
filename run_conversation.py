from agent.memory import load_memory
from agent.memory import save_memory
from agent.graph import agent_graph
import warnings
warnings.filterwarnings('ignore')


app = agent_graph()

USER_ID = "paciente_004"


def conversar(pergunta: str):
    historico = load_memory(USER_ID)

    result = app.invoke({
        "input": pergunta,
        "history": historico
    })

    resposta = result["resposta"]
    print(resposta)
    save_memory(USER_ID, "user", pergunta)
    save_memory(USER_ID, "agent", resposta)


conversar("O que é endometriose? E o que posso fazer para aliviar a dor?")
