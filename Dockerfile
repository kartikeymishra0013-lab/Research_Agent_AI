# ============================================================
# Scientific Document Intelligence Pipeline — Dockerfile
# Multi-stage build for lean production image
# ============================================================

# ─── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies into a prefix
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Scientific Document Intelligence Pipeline"
LABEL description="AI-powered pipeline for converting unstructured documents into structured knowledge"
LABEL version="2.0.0"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For PyMuPDF (PDF processing)
    libmupdf-dev \
    mupdf-tools \
    # For python-docx (Word documents)
    libxml2 \
    libxslt1.1 \
    # OCR: Tesseract + English language data
    tesseract-ocr \
    tesseract-ocr-eng \
    # OCR: pdf2image requires poppler utils (pdftoppm / pdftocairo)
    poppler-utils \
    # Networking & health checks
    curl \
    # Timezone support
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r pipeline && useradd -r -g pipeline -d /app -s /sbin/nologin pipeline

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create output directory with correct permissions
RUN mkdir -p data/input data/output && \
    chown -R pipeline:pipeline /app

# Switch to non-root user
USER pipeline

# Environment variables (overridable via docker-compose or .env)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# Health check (lightweight — just verifies imports)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.pipeline.orchestrator import PipelineOrchestrator; print('OK')" || exit 1

# Default command — can be overridden in docker-compose
CMD ["python", "-m", "src.main", "--help"]
