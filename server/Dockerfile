FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/main.py .

ENV MODEL_PATH=/app/models/final_q8_0.gguf
ENV MODEL_URL=https://github.com/leego972/titanai/releases/download/titan-1b-v1/final_q8_0.gguf
ENV N_CTX=4096
ENV N_THREADS=4

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
