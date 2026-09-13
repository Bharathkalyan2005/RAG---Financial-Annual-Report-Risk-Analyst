# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies and git for FlagEmbedding / pgvector
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default to running the FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
