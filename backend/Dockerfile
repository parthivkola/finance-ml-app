# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# System deps for psycopg2 and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create models directory
RUN mkdir -p /models

ENV PYTHONPATH=src
ENV MODELS_DIR=/models

EXPOSE 8080

# Run migrations then start server
CMD ["sh", "-c", "uv run alembic upgrade head && uv run fastapi run src/market_intelligence/api/main.py --port 8080 --host 0.0.0.0"]
