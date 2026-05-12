FROM python:3.10-slim

# Installeer systeem-tools die handig zijn voor Git en databases
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Zorg dat Python output direct naar de terminal gaat (handig voor logging)
ENV PYTHONUNBUFFERED=1

# Update pip
RUN pip install --upgrade pip
