# Multi-stage optimized Docker build
FROM python:3.11-slim AS builder

# Build stage - install dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Copy and consolidate requirements from all apps
COPY apps/api/requirements.txt ./api_requirements.txt
COPY apps/rag-script/requirements.txt ./rag_requirements.txt
COPY apps/ml-inference/requirements.txt ./ml_requirements.txt
RUN cat api_requirements.txt rag_requirements.txt ml_requirements.txt | sort | uniq > combined_requirements.txt && \
    pip wheel --no-cache-dir --no-deps --wheel-dir wheels -r combined_requirements.txt

# Production stage
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/apps/api:/app/apps/rag-script:/app/apps/ml-inference" \
    PORT=8000 \
    WORKERS=1

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client curl libpq5 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/combined_requirements.txt .
RUN pip install --no-cache-dir --find-links wheels -r combined_requirements.txt && \
    rm -rf /wheels combined_requirements.txt

# Copy application code
COPY --chown=appuser:appuser apps/ ./apps/
# Copy packages if they exist
COPY --chown=appuser:appuser packages ./packages
RUN mkdir -p data models uploads logs && chown -R appuser:appuser . && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

USER appuser

# Set Hugging Face cache directory
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/home/appuser/.cache/huggingface/datasets

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Direct command - no entrypoint script needed
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
