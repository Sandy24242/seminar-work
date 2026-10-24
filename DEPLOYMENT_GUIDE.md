# Dental Caries AI — 100% Free Cloud Deployment Guide

This guide provides step-by-step instructions to deploy the Dental Caries AI project publicly at **₹0 cost** (no paid subscriptions, no credit cards, no billing accounts, and zero risk of accidental charges).

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │             User / Browser             │
                      └───────────────────┬────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 │                                                 │
                 ▼                                                 ▼
     ┌────────────────────────┐                        ┌───────────────────────┐
     │  Vercel (Free Tier)    │                        │ Streamlit Cloud (Free)│
     │  React + Vite Frontend │                        │ Alternative UI Demo   │
     │  - Global Edge CDN     │                        │ - Questionnaire       │
     │  - Zero cold starts    │                        │ - Standalone app      │
     └───────────┬────────────┘                        └───────────────────────┘
                 │ REST API Calls
                 │ (/health, /models/status, /predict)
                 ▼
     ┌─────────────────────────────────────────────────────────┐
     │ Hugging Face Spaces (Docker SDK — Free Tier)            │
     │ - 2 vCPU · 16 GB RAM (No 512 MB Render OOM crashes)     │
     │ - FastAPI REST API (:8000)                              │
     │ - Native Git LFS hosting for 5 trained models (426 MB)  │
     │ - Automated wake-up detection & retry in React frontend │
     └─────────────────────────────────────────────────────────┘
```

---

## Part 1: Deploy Backend to Hugging Face Spaces (Free 16 GB RAM)

Hugging Face Spaces provides **2 vCPUs and 16 GB of RAM for free**, without requiring a credit card or billing account. This comfortably supports the TensorFlow runtime and all 5 models (~1.5 GB memory footprint) without Out-Of-Memory crashes.

### Step 1.1: Create a Free Hugging Face Account & Space
1. Sign up or log in at [huggingface.co](https://huggingface.co) (free, email only).
2. Click your profile icon (top right) → **New Space** (or go to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the Space settings:
   - **Space name**: e.g., `dental-caries-api`
   - **License**: `mit` or `apache-2.0`
   - **Select the Space SDK**: Choose **Docker** → **Blank**.
   - **Space hardware**: `CPU basic · 2 vCPU · 16 GB · Free` (default).
   - **Visibility**: **Public** (important: public visibility enables the browser frontend to contact the API without requiring secret authentication tokens).
4. Click **Create Space**.

### Step 1.2: Generate a Hugging Face Access Token
1. In Hugging Face, click your avatar → **Settings** → **Access Tokens**.
2. Click **Create new token**:
   - Token type: **Write**
   - Name: `github-actions-sync`
3. Copy the generated token (starts with `hf_...`).

### Step 1.3: Configure GitHub Secrets for Automatic Deployment
In your GitHub repository (`Sandy24242/seminar-work`):
1. Go to **Settings** → **Secrets and variables** → **Actions**.
2. Add the following repository secrets:
   - `HF_TOKEN`: Paste the Hugging Face Write token from Step 1.2.
   - `HF_SPACE`: Your Hugging Face Space repository path in the format `username/space-name` (for example: `Sandy24242/dental-caries-api`).
   - `BACKEND_URL`: `https://<username>-dental-caries-api.hf.space` (optional, used by the wake-up ping workflow).
3. Now, whenever you push code or merge to your deployment branch, GitHub Actions will automatically push the repository (including Git LFS weights) to Hugging Face Spaces, triggering a fresh Docker build.

### Step 1.4: Configure Backend Environment Variables in Hugging Face
1. In your Hugging Face Space, click **Settings** (tab near the top right).
2. Scroll down to **Variables and secrets**.
3. Under **Variables**, add:
   - Name: `DCAI_CORS_ORIGINS`
   - Value: `https://<your-vercel-app>.vercel.app,http://localhost:5173,http://127.0.0.1:5173`
   *(Note: The backend also automatically allows any `https://*.vercel.app` preview URL by default via regex).*

---

## Part 2: Deploy Frontend to Vercel (Free Tier)

Vercel hosts the React + Vite frontend globally with instant loading and zero cold starts.

### Step 2.1: Import Project into Vercel
1. Log in to [vercel.com](https://vercel.com) using your GitHub account (free, no credit card).
2. Click **Add New…** → **Project**.
3. Select your GitHub repository (`seminar-work`) and click **Import**.

### Step 2.2: Configure Build Settings
In the Vercel project configuration screen:
1. **Framework Preset**: Vite (automatically detected).
2. **Root Directory**: Click *Edit* and select `frontend`.
3. **Build & Development Settings**:
   - Build Command: `npm run build` (or leave default).
   - Output Directory: `dist` (or leave default).
4. **Environment Variables**:
   - Name: `VITE_API_BASE_URL`
   - Value: `https://<username>-<space-name>.hf.space` (replace with your Hugging Face Space direct URL, without a trailing slash).
5. Click **Deploy**.

Vercel will build the frontend in ~30 seconds and provide a production URL like `https://seminar-work.vercel.app`.

---

## Part 3: Deploy Alternative Streamlit UI (Streamlit Community Cloud)

If your professor or seminar panel also wants to test the original interactive Streamlit application with the oral questionnaire and downloadable PDF reports:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**.
3. Select:
   - **Repository**: `Sandy24242/seminar-work`
   - **Branch**: your deployment branch (e.g., `main` or `claude/gifted-lamport-nv8e4c`)
   - **Main file path**: `backend/streamlit_app.py`
   - **App URL**: choose an optional custom subdomain.
4. Click **Deploy!**.

*Note: Streamlit Community Cloud allocates 1 GB RAM per app. If running all five models pushes memory near the 1 GB boundary, Streamlit Cloud may display a resource warning.*

---

## Part 4: Handling Cold Starts & Free Monitoring

### How Cold Starts Are Handled
- Free cloud containers on Hugging Face Spaces sleep after a period of inactivity.
- When an asleep backend receives traffic, it wakes up and loads the models (~15–30 seconds).
- The React frontend includes **Adaptive Polling**:
  - If the backend is waking up, the UI displays a clean status card: *"Backend is waking up… Free-tier cloud containers sleep when idle."*
  - Automatically polls `/health` every 4 seconds with a live attempt counter and provides a manual **Retry Now** button.
  - As soon as the backend responds, the upload form is immediately enabled.

### Best-Effort Wake-Up Strategy (Free External Monitoring)
To keep the server warm during presentation hours:
1. **Cron-job.org (100% Free)**:
   - Register at [cron-job.org](https://cron-job.org).
   - Create a cron job pointing to `https://<your-space>.hf.space/health`.
   - Set execution schedule to every 10–14 minutes during class/presentation hours (e.g. 9 AM – 5 PM).
2. **GitHub Actions Scheduled Ping**:
   - The repository includes `.github/workflows/wake-up-ping.yml`, which automatically pings the backend every 20 minutes during weekdays when the secret `BACKEND_URL` is set.

*Reminder: External pings reduce the likelihood of cold starts, but free-tier hosts may still sleep or reboot for host maintenance. The application is built to handle cold starts gracefully regardless.*

---

## Part 5: End-to-End Verification Checklist

Verify each of the following before demonstrating to your professor:

- [ ] **PC Switched Off Test**: Open the Vercel URL on your mobile phone or another device connected to mobile data while your PC is completely turned off.
- [ ] **Backend Health Check**: Open `https://<username>-<space-name>.hf.space/health` in a browser. It should return `{"status":"ok",...}`.
- [ ] **Model Status**: Open `https://<username>-<space-name>.hf.space/models/status`. Check that `ready: true` and all 5 models show `"loaded": true`.
- [ ] **Image Upload & Inference**: Upload a test dental photograph on the Vercel website. Confirm that:
  - Stage 1 healthy/caries classification runs.
  - Lesion localizer marks bounding boxes.
  - Stage 3 G.V. Black and Stage 4 ICDAS predictions display.
  - Annotated image renders with colored bounding boxes.
