FROM python:3.11-slim

WORKDIR /app

# 🔥 Instala ffmpeg (ESSENCIAL pro WhatsApp)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Evita problemas com dependências nativas
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ⚠️ Evita problemas com asyncio dentro do container
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]