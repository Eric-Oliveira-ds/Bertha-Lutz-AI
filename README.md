# Bertha-Lutz-AI

## Visão Geral

Bertha-Lutz-AI é um agente conversacional de IA projetado para promover a saúde da mulher, abordando desafios comuns como sobrecarga cotidiana, lacunas em letramento em saúde, desinformação digital e barreiras no acesso ao cuidado médico. Inspirado na figura histórica Bertha Lutz, o projeto visa empoderar mulheres com informações precisas, acessíveis e personalizadas sobre saúde feminina.

### Problemas Abordados
- Mulheres postergam autocuidado devido à sobrecarga cotidiana, invisibilizando sua saúde.
- Lacunas de letramento em saúde dificultam entender rastreamento, periodicidade e sinais de alerta.
- Interpretação limitada de exames gera ansiedade e buscas inseguras online.
- Vergonha e tabus inibem relato de sintomas, exames e diálogo sobre sexualidade/ISTs.
- Desinformação digital e influenciadores não qualificados promovem condutas perigosas.
- Dificuldade de navegação no SUS/privado resulta em perda de tempo e acesso inadequado.
- Normalização de sintomas patológicos e negligência de saúde materna atrasam diagnósticos.
- Falta de acompanhamento longitudinal e baixa autonomia comprometem prevenção contínua.
- Violência obstétrica e prioridade nos outros afastam mulheres do cuidado.

## Solução

Bertha-Lutz-AI é um agente conversacional com Retrieval-Augmented Generation (RAG) oficial, memória persistente, guardrails médicos, avaliação automática e observabilidade completa. Ele oferece suporte personalizado, baseado em dados confiáveis, para consultas sobre saúde feminina, rastreamento preventivo e orientação segura.

## Funcionalidades

- **RAG Oficial**: Integração com fontes médicas autorizadas para respostas precisas e atualizadas.
- **Memória Persistente**: Mantém contexto de conversas para interações contínuas e personalizadas.
- **Guardrails Médicos**: Implementa barreiras éticas e de segurança para evitar conselhos inadequados.
- **Avaliação Automática**: Monitora e avalia a qualidade das respostas em tempo real.
- **Observabilidade Completa**: Ferramentas para rastreamento de desempenho, logs e métricas.
- **Suporte Multilíngue**: Disponível em português, com expansão futura.
- **Integração com SUS**: Orientações sobre navegação no sistema público de saúde brasileiro.

## Arquitetura

### Tecnologias Utilizadas
- **Linguagem**: Python
- **Framework de IA**: LangChain.
- **Banco de Dados**: PostgreSQL para memória persistente.
- **Infraestrutura**: Docker para containerização.
- **APIs**: FAST-API.
- **Monitoramento**: Prometheus e Grafana para observabilidade.

### Diagrama de Arquitetura
```
[Usuário] -> [Interface (Web/App)] -> [Agente IA (LangChain)] -> [RAG (Fontes Médicas)] -> [Memória (DB)] -> [Guardrails (Validação)] -> [Avaliação (Métricas)]
```

## Instalação

### Pré-requisitos
- Python 3.8+
- Docker

### Passos
1. Clone o repositório:
    ```bash
    git clone https://github.com/Eric-Oliveira-ds/Bertha-Lutz-AI.git
    cd Bertha-Lutz-AI
    ```

2. Instale dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure variáveis de ambiente (ex.: API keys, DB credentials) em `.env`.

4. Execute localmente:
    ```bash
    docker-compose up
    ```

5. Para deploy em produção, use Kubernetes ou serviços gerenciados.

## Uso

1. Acesse a interface web ou app móvel.
2. Inicie uma conversa digitando sintomas ou perguntas sobre saúde feminina.
3. O agente responde com orientações baseadas em dados oficiais, lembrando interações anteriores.
4. Para desenvolvedores: Use a API REST para integrações.

### Exemplo de Interação
- Usuário: "Quais exames devo fazer anualmente?"
- Bertha-Lutz-AI: "Baseado em diretrizes do Ministério da Saúde, mulheres acima de 25 anos devem fazer Papanicolau e mamografia regularmente. Consulte um profissional para personalização."

## Desenvolvimento e Contribuição

### Estrutura do Projeto
```
Bertha-Lutz-AI/
├── src/                 # Código fonte
│   ├── agent/           # Lógica do agente IA
│   ├── rag/             # Componente RAG
│   └── guardrails/      # Validações médicas
├── tests/               # Testes unitários e de integração
├── docs/                # Documentação adicional
├── docker/              # Configurações Docker
└── README.md            # Este arquivo
```

### Como Contribuir
1. Fork o repositório.
2. Crie uma branch para sua feature: `git checkout -b feature/nova-funcionalidade`.
3. Faça commits claros e testes.
4. Abra um Pull Request com descrição detalhada.

### Webhook
http://api:8000/webhook/whatsapp

### Testes
Execute testes com:
```bash
pytest
```

## Métricas e Avaliação

- **Precisão**: >95% em respostas baseadas em fontes oficiais (medido via avaliação automática).
- **Engajamento**: Tempo médio de sessão >5 minutos.
- **Segurança**: 0% de violações de guardrails em testes simulados.

## Licença

Este projeto é licenciado sob MIT. Veja [LICENSE](LICENSE) para detalhes.

## Autores

- **Eric Oliveira** (Eric-Oliveira-ds) - Desenvolvedor Principal.

## Contato

Para dúvidas ou colaborações: eric.oliveira.pro@gmail.com ou abra uma issue no GitHub.

---

*Bertha-Lutz-AI: Empoderando a saúde da mulher com IA responsável.*