# ---------------------------------------------------------------------------
# Backend image. Multi-stage so the CUDA-free torch wheels are resolved once
# and the final image carries no build tooling.
#
#   docker build -t oral-cancer-api ./backend
#   docker run -p 8000:8000 -v "$(pwd)/backend/models:/app/models" oral-cancer-api
#
# The weights are NOT baked in -- mount them, or set MODEL_URL to fetch at boot.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .

# Installed into a venv so the whole tree copies cleanly into the runtime stage.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp1 is required by torch; the rest of the toolchain is not.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Run unprivileged: this service accepts file uploads from the network.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/EfficientNetB0_Dataset02_OralCancer_EXP4A.pkl

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
RUN mkdir -p /app/models && chown appuser:appuser /app/models

USER appuser
EXPOSE 8000

# Reports unhealthy while the model is missing or failed to load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,json,os,sys; \
u=f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/api/health'; \
sys.exit(0 if json.load(urllib.request.urlopen(u))['model_loaded'] else 1)"

# Shell form so $PORT is expanded by the platform at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
