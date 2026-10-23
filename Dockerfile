# Dental Caries AI — backend (FastAPI + TensorFlow inference), with the trained models baked in.
#
# Build from the REPOSITORY ROOT (the image needs backend/ and models/):
#   git lfs install && git lfs pull          # the .keras files must be real weights, not LFS pointer files
#   docker build -t dental-caries-api .
#   docker run --rm -p 8000:8000 -e DCAI_CORS_ORIGINS=https://<your-frontend>.vercel.app dental-caries-api
#
# Hosts that set $PORT (Render, Railway, Fly.io, Hugging Face Spaces) are supported. The models need ~1.5 GB RAM.
# To build without weights and mount them at run time instead:
#   docker build --build-arg REQUIRE_MODELS=0 -t dental-caries-api .
#   docker run -p 8000:8000 -v /path/to/models:/app/models:ro dental-caries-api

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DCAI_MODELS_DIR=/app/models \
    PORT=8000

# Non-root user (uid 1000 is also what Hugging Face Spaces expects)
RUN useradd --create-home --uid 1000 app
WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app backend/ /app/backend/
COPY --chown=app:app models/ /app/models/

# Fail the build early if the weights are Git LFS pointer files (run `git lfs pull` before building).
ARG REQUIRE_MODELS=1
RUN if [ "$REQUIRE_MODELS" = "1" ]; then python scripts/check_weights_present.py /app/models; fi

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# One worker: each worker loads all five models (~1.5 GB).
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
