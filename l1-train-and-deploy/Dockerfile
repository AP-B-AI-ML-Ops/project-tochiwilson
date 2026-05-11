# Gebruik een lichte versie van Python
FROM python:3.12-slim

# Maak een mapje genaamd /app aan in de container en ga daar in staan
WORKDIR /app

# Update pip (pakketbeheerder)
RUN pip install -U pip

# Kopieer EERST de requirements.txt en installeer deze
# (Dit is een slimme truc: als je code later wijzigt, hoeft Docker niet alle packages opnieuw te downloaden)
COPY requirements.txt .
RUN pip install -r requirements.txt --upgrade

# Kopieer nu de rest van je python scripts naar de /app map
COPY . .

# Vertel dit script waar de Prefect en MLflow servers draaien
# (Dit matcht met de namen uit je docker-compose.yml!)
ENV PREFECT_API_URL=http://orchestration:4200/api
ENV MLFLOW_TRACKING_URI=http://experiment-tracking:5000

# Als de container start, run dan de master flow
CMD [ "python", "main.py" ]