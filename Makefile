.PHONY: help build up down train serve batch test lint all

help:
	@echo "Beschikbare commando's:"
	@echo "  make build    - Bouw alle Docker images"
	@echo "  make up       - Start de infrastructuur (database, mlflow, prefect, grafana)"
	@echo "  make down     - Stop alle containers"
	@echo "  make train    - Train het model"
	@echo "  make serve    - Start de web service"
	@echo "  make batch    - Start de batch service (scheduled)"
	@echo "  make test     - Run unit tests"
	@echo "  make lint     - Run pre-commit hooks"
	@echo "  make all      - Start database, MLflow, Prefect, Grafana en batch-service"

build:
	docker compose build

up:
	docker compose up database experiment-tracking orchestration grafana

down:
	docker compose down

train:
	docker compose --profile training run --rm train-deploy

serve:
	docker compose up web-service

batch:
	docker compose up batch-service

test:
	pytest tests/ -v

lint:
	pre-commit run --all-files

all:
	docker compose up database experiment-tracking orchestration grafana batch-service
