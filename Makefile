.PHONY: help build up down train serve batch test lint

help:
	@echo "Beschikbare commando's:"
	@echo "  make build    - Bouw alle Docker images"
	@echo "  make up       - Start de infrastructuur (database, mlflow, prefect, grafana)"
	@echo "  make down     - Stop alle containers"
	@echo "  make train    - Train het model"
	@echo "  make serve    - Start de web service"
	@echo "  make batch    - Run de batch service"
	@echo "  make test     - Run unit tests"
	@echo "  make lint     - Run pre-commit hooks"

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
	docker compose --profile batch run --rm batch-service

test:
	pytest tests/ -v

lint:
	pre-commit run --all-files
