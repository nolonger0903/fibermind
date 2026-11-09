.PHONY: install dev test lint generate-intents train-intent evaluate evaluate-ragas sweep docker-up docker-down

install:
	python -m pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

test:
	pytest -q

lint:
	ruff check .

generate-intents:
	python scripts/generate_intent_dataset.py

train-intent:
	python scripts/train_intent_bert.py --epochs 3

evaluate:
	python evaluation/run_evaluation.py

evaluate-ragas:
	python evaluation/run_ragas.py

sweep:
	python evaluation/run_retrieval_sweep.py

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
