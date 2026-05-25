FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN prefect config set PREFECT_API_URL=http://orchestration:4200/api

COPY batch.py .
COPY retrain.py .
COPY startWorker.sh .

RUN chmod +x startWorker.sh

CMD ["./startWorker.sh"]
