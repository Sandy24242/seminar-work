# FastAPI backend + the 5 real .keras models, for Hugging Face Spaces (Docker SDK) or any container host.
# Build from a checkout where `git lfs pull` has run, so models/*.keras are real weights, not LFS pointers.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 DCAI_MODELS_DIR=/app/models PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY --chown=user backend backend
COPY --chown=user models models
# Fail the build if LFS pointers were copied instead of weights
RUN python -c "import glob,os,sys; s=[f for f in glob.glob('models/*.keras') if os.path.getsize(f)<1_000_000]; sys.exit('LFS pointer files, run git lfs pull: %s'%s if s or len(glob.glob('models/*.keras'))!=5 else 0)"
USER user
WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
