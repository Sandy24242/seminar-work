# Deployment: frontend on Vercel, backend on Hugging Face Spaces

## Why the backend is not on Vercel
TensorFlow 2.21 alone is ~1.3 GB installed and the five models are ~426 MB, against Vercel's
500 MB (uncompressed) Python function bundle limit. Loading all models needs ~1.6 GB RAM and a cold
start of tens of seconds. So the FastAPI backend runs as a Docker container instead (free Hugging Face
Spaces CPU tier: 2 vCPU / 16 GB RAM). Inference is real; nothing is mocked.

## 1. Backend → Hugging Face Space (Docker)
1. huggingface.co → New Space → SDK **Docker** (blank), hardware **CPU basic (free)**.
2. From a checkout of this repo with real weights (`git lfs install && git lfs pull`):
   ```bash
   git clone https://huggingface.co/spaces/<you>/<space> hf-space
   cp -r Dockerfile .dockerignore backend models hf-space/
   cd hf-space && git lfs install && git lfs track "*.keras" && git add . 
   printf -- '---\ntitle: Dental Caries AI API\nsdk: docker\napp_port: 7860\n---\n' > README.md && git add README.md
   git commit -m "Deploy API" && git push   # use an HF write token as the password
   ```
3. Space → Settings → Variables: `DCAI_CORS_ORIGINS=https://<your-project>.vercel.app` and optionally
   `DCAI_CORS_ORIGIN_REGEX=https://<your-project>-.*\.vercel\.app` for previews.
4. Check `https://<you>-<space>.hf.space/models/status` → `"ready": true`.

## 2. Frontend → Vercel
Vercel → Add New Project → import `Sandy24242/seminar-work`, branch as production branch.
- Root Directory: `frontend` (build/output come from `frontend/vercel.json`: `npm run build`, `dist`)
- Environment variable: `VITE_API_URL=https://<you>-<space>.hf.space` (no trailing slash), then Deploy.
SPA routes are rewritten to `index.html` by `frontend/vercel.json`.
