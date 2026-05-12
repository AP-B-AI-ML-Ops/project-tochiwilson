FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16.0 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://admin:admin@database:5432/mlflow_db", \
     "--default-artifact-root", "/mlruns"]