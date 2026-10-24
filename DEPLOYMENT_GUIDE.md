# Dental Caries AI — Zero-Cost Cloud Deployment Guide

This guide details how to deploy the Dental Caries AI semester project for **₹0** (no paid subscriptions, no credit cards, no billing accounts, and zero risk of charges).

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │             User / Browser             │
                      │         (Professor, Panel, Mobile)     │
                      └───────────────────┬────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 │                                                 │
                 ▼                                                 ▼
     ┌────────────────────────┐                        ┌───────────────────────┐
     │  Vercel (Optional)     │                        │ Streamlit Cloud (Core)│
     │  Project Landing Page  │                        │ Live Cloud Application│
     │  - Fast Global Edge CDN│ ──── "Launch Demo" ───►│ - All 5 models loaded │
     │  - Architecture overview│                       │ - 4-stage ML cascade  │
     │  - Developer API toggle│                        │ - Questionnaire & PDF │
     └────────────────────────┘                        └───────────────────────┘
```

---

## 1. Primary Deployment: Streamlit Community Cloud

The entire Dental Caries AI application (4-stage multi-CNN cascade, interactive 7-question oral-habits assessment, detection visualizer, and downloadable PDF reports) runs directly on **Streamlit Community Cloud** with **zero cost** and remains accessible even when your PC is switched off.

### Step 1.1: Push Your Code and Models to GitHub
1. Ensure your latest changes on branch `claude/gifted-lamport-nv8e4c` (or merged to `main`) are pushed to GitHub:
   ```powershell
   git push origin claude/gifted-lamport-nv8e4c
   ```
2. Verify in your GitHub repository (`Sandy24242/seminar-work`) that the `models/` directory contains all 5 `.keras` model files and their metadata `.json` files.

### Step 1.2: Connect to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and click **Sign in with GitHub** (free, no credit card required).
2. Click **Create app** (or **New app**).
3. Fill in the repository settings:
   - **Repository**: `Sandy24242/seminar-work`
   - **Branch**: `claude/gifted-lamport-nv8e4c` (or `main`)
   - **Main file path**: `backend/streamlit_app.py`
   - **App URL (optional)**: e.g. `dental-caries-ai` (results in `https://dental-caries-ai.streamlit.app`)
4. Click **Deploy!**.

### Step 1.3: How Dependencies and Models Are Resolved
- **Dependencies**: Streamlit Cloud automatically detects `backend/requirements.txt` (and the root `requirements.txt`), installing `streamlit`, `plotly`, `tensorflow==2.21.0`, `keras==3.13.2`, `reportlab`, `opencv-python-headless`, and `numpy`.
- **Model Files**: The application's `config.resolve_models_dir()` automatically checks `models/` at the repository root and loads the five trained Keras models into memory.
- **Resource Caching**: Loaded models are cached in memory using `@st.cache_resource`, ensuring that once loaded, subsequent predictions run fast without reloading weights.

---

## 2. Optional Deployment: Vercel Landing Page

The React + Vite frontend can be deployed to Vercel as a fast, polished project landing page that explains the system architecture and links directly to the live Streamlit demo.

### Step 2.1: Import Project to Vercel
1. Log in to [vercel.com](https://vercel.com) using your GitHub account (Hobby Free tier; no credit card required).
2. Click **Add New…** → **Project** and select `seminar-work`.
3. In the project settings:
   - **Root Directory**: Click *Edit* and choose `frontend`.
   - **Framework Preset**: Vite (automatically detected).
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Under **Environment Variables**, add:
   - `VITE_STREAMLIT_URL`: The public URL of your deployed Streamlit application (e.g. `https://dental-caries-ai.streamlit.app`).
5. Click **Deploy**.

The Vercel landing page will deploy in under 1 minute with global CDN caching.

---

## 3. Verification Checklist

Before presenting to your professor, verify the following:

- [ ] **PC Switched Off Test**: Open your Streamlit public URL (`https://<your-app>.streamlit.app`) from a mobile phone or another computer while your personal PC is powered off.
- [ ] **Home Page**: Verify that the landing page renders with the project summary, stage workflow diagram, and navigation sidebar.
- [ ] **Oral Habits Questionnaire**: Navigate to **Assessment**, fill out the 7 lifestyle questions, and confirm risk scoring (LOW / MODERATE / HIGH).
- [ ] **Image Upload & Inference**: Upload a sample intraoral photograph and click **Analyze**. Verify:
  - Stage 1 caries screening runs and outputs probability/threshold.
  - Stage 2 FDI tooth numbering detects tooth positions.
  - Cavity localizer identifies lesion bounding boxes.
  - Stage 3 G.V. Black (Classes I–VI) and Stage 4 ICDAS (1–6) scores are assigned.
  - Visual overlay renders detection boxes on the photograph.
- [ ] **Report Generation**: Click **Download Clinical Report (PDF)** to verify that ReportLab generates a valid, downloadable PDF report.

---

## 4. Fallback Cloud Demonstration (Google Colab)

If Streamlit Community Cloud experiences temporary provider downtime or container sleep during your presentation:

1. Open [`backend/colab/Run_Dental_Caries_AI_Website.ipynb`](file:///c:/Users/abuna/Desktop/proj/seminar-work/backend/colab/Run_Dental_Caries_AI_Website.ipynb) in [Google Colab](https://colab.research.google.com).
2. Run Steps 1–6 to mount Google Drive and install packages.
3. Run Step 7 (Option B: Cloudflare Quick Tunnel) to generate a public demonstration URL that runs on Google Colab's free GPU/CPU runtime.
