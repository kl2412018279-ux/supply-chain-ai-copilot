.PHONY: install install-dev data train ingest api app test test-all lint format eval up down

install:        ## full install (agent, embeddings, MLflow)
	pip install -r requirements.txt
install-dev:    ## lightweight install (what CI uses)
	pip install -r requirements-ci.txt
data:           ## regenerate synthetic container-volume CSV
	python -m src.forecasting.data
train:          ## train LightGBM with grid search + MLflow tracking
	python -m src.forecasting.train --tune --mlflow
ingest:         ## embed documents in data/raw into the vector store
	python -m src.rag.ingest_pdf
eval:           ## retrieval quality report (hit@k / MRR)
	python -m src.rag.evaluate --k 3
api:
	uvicorn src.api.main:app --reload --port 8000
app:
	streamlit run app.py
test:           ## unit tests (no services needed)
	pytest
test-all:       ## unit + integration tests against the docker-compose services
	TEST_DATABASE_URL=postgresql://admin:password123@localhost:5432/supply_chain \
	TEST_REDIS_URL=redis://localhost:6379/15 pytest
lint:
	ruff check . && ruff format --check .
format:
	ruff check --fix . && ruff format .
up:
	docker compose up -d --build
down:
	docker compose down
