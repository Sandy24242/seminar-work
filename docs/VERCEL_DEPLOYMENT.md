# Deploying the backend on Vercel

## Why the deployment failed

`[LAMBDA_SIZE_EXCEEDED] Total bundle size (2464.21 MB) exceeds the maximum function size (500 MB).`

The 500 MB cap is Vercel's standard limit for Python functions. Measured locally (Linux, `pip install --target`):

| What | Size |
|---|---|
| Old `requirements.txt` (incl. Streamlit, Plotly, pyarrow) | 2742 MB |
| API-only `requirements.txt` (this commit) | 2382 MB |
| …minus `tensorflow/include` (C++ headers, not used at runtime) | 1827 MB |
| `tensorflow` alone (`libtensorflow_cc.so.2` is 1046 MB) | ~1450 MB |
| Trained models `models/*.keras` (Git LFS) | 427 MB |

TensorFlow by itself is about three times the 500 MB cap. **No safe trimming gets this backend under 500 MB.**
The only way to run it on Vercel is **Large Functions** (beta, up to 5 GB). The API-only dependency set was verified
with the real weights. All five models loaded and `/predict` returned a real cascade result. The API tests passed (38).

## Large Functions: requirements (as reported by Vercel's changelog and KB; re-check in the dashboard)

- Requires **Fluid Compute** with **Active CPU** enabled on the project.
- Opt in with the environment variable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`. New projects may already be enrolled.
- Not available with Secure Compute or Static IPs.
- On the **Hobby** plan, a community report says the 500 MB cap still applied. Hobby eligibility is unconfirmed.

### Manual steps (Vercel dashboard, Project → Settings)

1. **Functions → Fluid Compute**: enabled. Make sure **Active CPU** pricing/compute is on.
2. **Environment Variables**: add `VERCEL_SUPPORT_LARGE_FUNCTIONS` = `1` for **Production** and **Preview**.
3. **Git → Git LFS**: enable it. Otherwise the clone contains 133-byte pointer files instead of the `.keras` weights,
   and `/models/status` reports every model as unavailable.
4. **General → Root Directory**: the models live in `models/` at the repository root. The backend looks for them in
   `$DCAI_MODELS_DIR`, then in `backend/models`, then in `../models`. If Root Directory is `backend`, confirm that
   the deployed function actually contains `../models/*.keras`. After deploying, call `GET /models/status`. Every model
   must show `"loaded": true`.
5. Redeploy. Then check `GET /health`, `GET /models/status` and a real `POST /predict`.

Cold starts load about 430 MB of models plus TensorFlow. Keep the function memory and `maxDuration` high enough
(a local load takes about 12 s). Larger memory and duration can cost more, so check your plan first.

## Fallback: a separate backend host

Use this if Large Functions is unavailable on your plan, or if the deployment still fails.

1. Keep the React frontend on Vercel (`frontend/`, a static Vite build). Set its API URL (`frontend/.env.example`) to the backend URL.
2. Run the FastAPI backend as a container on a host with ≥ 2 GB RAM and no small bundle limit. Examples:
   Google Cloud Run, Render, Railway, Fly.io, or a Hugging Face Space (Docker).
   Build steps: `git lfs pull`, `pip install -r backend/requirements.txt`,
   `uvicorn api.main:app --host 0.0.0.0 --port $PORT` (from `backend/`).
3. Allow the frontend origin in the API's CORS settings, then verify `/models/status` and `/predict` again.
Pick a free tier or get approval before anything that bills.
