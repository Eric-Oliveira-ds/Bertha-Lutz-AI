# Bertha-Lutz-AI

## Visão Geral

Bertha-Lutz-AI é um agente conversacional de IA projetado para promover a saúde da mulher, abordando desafios comuns como sobrecarga cotidiana, lacunas em letramento em saúde, desinformação digital e barreiras no acesso ao cuidado médico. Inspirado na figura histórica Bertha Lutz, o projeto visa empoderar mulheres com informações precisas, acessíveis e personalizadas sobre saúde feminina. A interação acontece via **WhatsApp** e **web**, com voz (fala e escuta).

### Problemas Abordados
- Mulheres postergam autocuidado devido à sobrecarga cotidiana, invisibilizando sua saúde.
- Lacunas de letramento em saúde dificultam entender rastreamento, periodicidade e sinais de alerta.
- Interpretação limitada de exames gera ansiedade e buscas inseguras online.
- Vergonha e tabus inibem relato de sintomas, exames e diálogo sobre sexualidade/ISTs.
- Desinformação digital e influenciadores não qualificados promovem condutas perigosas.
- Dificuldade de navegação no SUS/privado resulta em perda de tempo e acesso inadequado.
- Normalização de sintomas patológicos e negligência de saúde materna atrasam diagnósticos.
- Falta de acompanhamento longitudinal compromete a prevenção contínua.

## Solução

Bertha-Lutz-AI é um **agente de IA com fluxo orquestrado por LangGraph**, com **RAG (Retrieval-Augmented Generation)** baseado em documentos oficiais, **memória persistente** em PostgreSQL, **roteamento clínico inteligente**, **guardrails médicos**, **avaliação automática com DeepEval** e **observabilidade completa (Prometheus + Grafana)**. As respostas são enviadas por WhatsApp (texto ou áudio) com suporte a voo/fala.

## Funcionalidades

- **Interação via WhatsApp**: integração com Evolution API para envio/recepção de mensagens de texto e áudio.
- **Voz de ponta a ponta**: Reconhecimento de fala (faster-whisper) e síntese de voz em português (edge-tts).
- **RAG oficial**: recuperação de contexto em fontes médicas autorizadas (PDFs do Ministério da Saúde, Fedor, etc.).
- **Roteamento clínico inteligente**: o supervisor decide entre coleta de dados, resposta geral, follow-up, revisão humana ou rota de alto risco.
- **Memória Persistente**: PostgreSQL para contexto de conversas longitudinais.
- **Guardrails Médicos**: auditoria de segurança que bloqueia respostas com nomes de medicamentos e encaminha para o médico.
- **Avaliação Automática**: DeepEval (faithfulness e relevância) com métricas expostas via Prometheus.
- **Observabilidade Completa**: Prometheus + Grafana para rastreamento de desempenho, custo, latência e blocagens.
- **Acompanhamento**: agendamento automático de follow-ups por WhatsApp.

## Arquitetura

### Fluxo do Agente (LangGraph)

```
[WhatsApp/Web] -> [API FastAPI] -> [Webhook] -> [STT (áudio)] -> [Agent Graph (LangGraph)]
    -> [supervisor] -> [RAG (Chroma)] -> [collector | general | risk | followup | human_review]
    -> [guardrails] -> [TTS (áudio)] -> [Resposta via Evolution API]
```

Nós do grafo (em `agent/nodes/`):
- **supervisor**: rota clínica principal, classifica risco, define provedor LLM.
- **coleta (collector)**: extrai dados clínicos estruturados (idade, gestação, citologia, etc.) e persiste perfil.
- **general**: respostas amigáveis ou conversa sobre exames, sem diagnóstico/prescrição.
- **risk**: orientação para situações de risco alto (nunca com diagnóstico).
- **followup**: agenda visitas de acompanhamento.
- **human_review**: fila para revisão humana quando necessário.
- **guardrails**: auditoria final de segurança da resposta.

### Tecnologias Utilizadas

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Orquestração | LangChain, LangGraph |
| LLMs | OpenAI (gpt-4o-mini, text-embedding-3-small) e Groq (gpt-oss-120b) |
| RAG | ChromaDB + LangChain, persistência local (`chroma_db/`) |
| Banco de dados | PostgreSQL 15 (SQLAlchemy + psycopg 3) |
| Fala | faster-whisper (STT), edge-tts (TTS) |
| WhatsApp | Evolution API |
| Observabilidade | Prometheus + Grafana (dashboards 14 painel) |
| Avaliação | DeepEval |
| Agendamento | APScheduler |
| Frontend | React 19, Vite, axios |

## Estrutura do Projeto

```
Bertha-Lutz-AI/
├── agent/                   # Agente IA (LangGraph)
│   ├── graph.py             # Definição do grafo de estados
│   ├── state.py             # Estado do agente (AgentState)
│   ├── router.py            # Roteamento entre nós
│   ├── llm.py / providers/  # Fábrica de LLMs (OpenAI/Groq)
│   ├── guardrails.py        # Auditoria de segurança da resposta
│   ├── rag.py               # Ingestão de PDFs no Chroma
│   ├── nodes/               # Nós do grafo
│   ├── services/            # risk_engine, clinical_db, followup_scheduler
│   ├── memory/              # Memória persistente (PostgreSQL)
│   ├── metrics/             # Métricas Prometheus customizadas
│   └── tools/               # search_protocol (RAG), output_parser
├── api/                     # Backend FastAPI
│   ├── main.py              # App, endpoints, webhook, agendador
│   ├── stt.py / tts.py      # Transcrição e síntese de voz
│   ├── send_message.py      # Envio via Evolution API
│   ├── validar_cpf.py, normalizar_telefone.py
│   └── db_create_tables.py  # Criação das tabelas (espelha sql/)
├── bertha-lutz-front/       # Frontend React (Vite) com Landing + Cadastro
├── monitoring/              # Prometheus e Grafana (provisioning + dashboard)
├── pdf/                     # Fontes oficiais (PDFs de saúde da mulher)
├── prompts/                 # Prompt base do agente (agent_pai.txt)
├── sql/                     # Esquema SQL das tabelas
├── chroma_db/               # Vetores persistidos do RAG
├── test_dataset.py          # Avaliação DeepEval (faithfulness/relevância)
├── docker-compose.yml       # api, postgres, chroma, prometheus, grafana, evolution, redis
└── Dockerfile
```

## Instalação

### Pré-requisitos
- Python 3.11+
- Docker e Docker Compose
- Node.js (18+) e npm (para o frontend)
- Chaves de API OpenAI e/ou Groq

### Passos

1. Clone o repositório:
    ```bash
    git clone https://github.com/Eric-Oliveira-ds/Bertha-Lutz-AI.git
    cd Bertha-Lutz-AI
    ```

2. Crie um arquivo `.env` a partir das variáveis esperadas (ver `docker-compose.yml` e o código). Importante: **nunca commite `.env`** — ele contém segredos.

3. Suba a infraestrutura e o backend:
    ```bash
    docker-compose up --build
    ```
    A API fica em `http://localhost:8000`, o Prometheus em `:9090` e o Grafana em `:3000` (admin/admin).

4. Inicie o frontend (em outro terminal):
    ```bash
    cd bertha-lutz-front
    npm install
    npm run dev
    ```
    Acesse `http://localhost:5173`.

5. (Opcional) Re-ingestão dos PDFs no RAG:
    ```bash
    python -m agent.rag
    ```

6. (Opcional) Rodar a avaliação automática:
    ```bash
    python test_dataset.py
    ```

## Uso

- **Frontend (web)**: página de captura → vídeo de apresentação → cadastro (nome, CPF, data de nascimento, telefone).
- **WhatsApp**: configure uma instância na Evolution API (porta 8080) e aponte o webhook para o servidor. O agente responde em texto e áudio.
- **API REST**: consulte os endpoints abaixo para integrações.

### Endpoints principais

| Método | Rota | Descrição |
|---|---|---|
| POST | `/register` | Cadastro de usuária (valida CPF/data) e envio de boas-vindas via WhatsApp |
| POST | `/webhook/whatsapp` | Webhook do Evolution API (recebe texto e áudio) |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/metrics/evaluation` | Atualiza scores de avaliação (faithfulness/relevância) |

## Testes e Avaliação

Não há suíte pytest; a avaliação é feita com **DeepEval**:

```bash
python test_dataset.py
```

O script avalia **faithfulness** e **answer relevancy** das respostas do agente (com GPT-4o-mini como juiz) para casos de exame preventivo, endometriose e antibióticos na gestação, e publica os scores nos gauges do Prometheus.

## Observabilidade

- **Prometheus** (porta 9090): rastreia `api:8000`.
- **Grafana** (porta 3000): dashboard pronto com painéis de HTTP RPS, latência de LLM, tokens (in/out), custo estimado, blocagens de guardrails, uso de CPU/RAM, RPS por endpoint e scores de qualidade do RAG.

## Contribuição

1. Fork o repositório.
2. Crie uma branch para sua feature: `git checkout -b feature/nova-funcionalidade`.
3. Faça commits claros e mantenha os padrões existentes.
4. Abra um Pull Request com descrição detalhada.

## Licença

Este projeto é licenciado sob a **Apache License 2.0**. Veja [LICENSE](LICENSE) para detalhes.

## Autores

- **Eric Oliveira** (Eric-Oliveira-ds) - Desenvolvedor Principal.

## Contato

Para dúvidas ou colaborações: eric.oliveira.pro@gmail.com ou abra uma issue no GitHub.

---

*Bertha-Lutz-AI: Empoderando a saúde da mulher com IA responsável.*