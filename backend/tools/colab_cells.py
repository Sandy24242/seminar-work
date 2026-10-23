"""
Single source of truth for the Google Colab cells.
Used by tools/make_colab_notebook.py (-> colab/Run_Dental_Caries_AI_Website.ipynb) and tools/make_guide.py
(-> docs/Dental_Caries_AI_Colab_Guide.pdf), so the notebook and the PDF guide always contain the same commands.
Each entry: (step id, title, code).
"""

CELL_MOUNT = '''# STEP 1 — Check the runtime and mount Google Drive
import os, subprocess
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
print(gpu or "No GPU (CPU runtime) - the app still works, just slower")
from google.colab import drive
drive.mount("/content/drive")
print("Drive mounted:", os.path.isdir("/content/drive/MyDrive"))'''

CELL_PATHS = '''# STEP 2 — Locate the website code and the trained models on Google Drive
import os, glob
DRIVE_PROJECT = "/content/drive/MyDrive/dentalcaries"              # project folder (or shortcut)
MODELS_DRIVE  = f"{DRIVE_PROJECT}/pipeline_stage1_to_4_corrected"  # written by the notebook
REPO_URL = "https://github.com/Sandy24242/seminar-work.git"        # preferred source of the website code
ZIP_CANDIDATES = [f"{DRIVE_PROJECT}/Dental_Caries_AI_Website.zip",  # fallback if GitHub cannot be reached
                  "/content/drive/MyDrive/Dental_Caries_AI_Website.zip",
                  "/content/Dental_Caries_AI_Website.zip"]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.isfile(p)), None)
print("Website code:", REPO_URL, "| fallback ZIP:", ZIP_PATH or "none")
print("Model folder:", MODELS_DRIVE, "->", "found" if os.path.isdir(MODELS_DRIVE) else "NOT FOUND")
for f in sorted(glob.glob(f"{MODELS_DRIVE}/*.keras")):
    print(f"   {os.path.basename(f):34} {os.path.getsize(f)/1e6:7.1f} MB")'''

CELL_EXTRACT = '''# STEP 3 — Put the website on the fast local Colab disk (GitHub clone, or the ZIP as a fallback)
import os, zipfile, shutil, subprocess
for d in ("/content/seminar-work", "/content/Dental_Caries_AI_Website"):
    shutil.rmtree(d, ignore_errors=True)
r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/seminar-work"], capture_output=True, text=True)
if r.returncode == 0:
    APP_DIR = "/content/seminar-work/backend"
    print("cloned", REPO_URL)
else:
    print("git clone failed:", r.stderr.strip()[-300:])
    assert ZIP_PATH, "No website ZIP either - upload Dental_Caries_AI_Website.zip to MyDrive/dentalcaries/ and run Step 2 again"
    with zipfile.ZipFile(ZIP_PATH) as z:
        bad = z.testzip()
        assert bad is None, f"The ZIP is damaged (first bad file: {bad}) - upload it again"
        z.extractall("/content")
    APP_DIR = "/content/Dental_Caries_AI_Website"
ENTRY = "streamlit_app.py" if os.path.exists(f"{APP_DIR}/streamlit_app.py") else "app.py"   # app.py = older ZIPs
for f in [ENTRY, "config.py", "requirements-colab.txt", "pipeline/runner.py", "assets/logo.png"]:
    print(("OK      " if os.path.exists(f"{APP_DIR}/{f}") else "MISSING ") + f)
%cd $APP_DIR'''

CELL_INSTALL = '''# STEP 4 — Install the two extra packages (TensorFlow/Keras are already in Colab)
!pip install -q -r {APP_DIR}/requirements-colab.txt
import tensorflow as tf, keras, streamlit, reportlab
print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
print("Streamlit", streamlit.__version__, "| ReportLab", reportlab.Version)'''

CELL_MODELS = '''# STEP 5 — Copy the trained models next to the app and check that they load
import os, shutil, glob
LOCAL_MODELS = f"{APP_DIR}/models"
os.makedirs(LOCAL_MODELS, exist_ok=True)
for f in glob.glob(f"{MODELS_DRIVE}/*.keras") + glob.glob(f"{MODELS_DRIVE}/*_metadata.json"):
    dst = f"{LOCAL_MODELS}/{os.path.basename(f)}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(f)):
        shutil.copy(f, dst); print("copied", os.path.basename(f))
os.environ["DCAI_MODELS_DIR"] = LOCAL_MODELS
!cd {APP_DIR} && python scripts/check_models.py "$DCAI_MODELS_DIR"'''

CELL_LAUNCH = '''# STEP 6 — Start the Streamlit website in the background
import subprocess, time, urllib.request, os
subprocess.run("pkill -f 'streamlit run' || true", shell=True)
log = open("/content/streamlit.log", "w")
cmd = ["streamlit", "run", ENTRY, "--server.port", "8501", "--server.address", "0.0.0.0",
       "--server.headless", "true", "--server.enableCORS", "false",
       "--server.enableXsrfProtection", "false"]
env = {**os.environ, "DCAI_MODELS_DIR": LOCAL_MODELS}
proc = subprocess.Popen(cmd, cwd=APP_DIR, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
HEALTH = "http://localhost:8501/_stcore/health"
for _ in range(60):
    try:
        if urllib.request.urlopen(HEALTH, timeout=2).read() == b"ok":
            print("Streamlit is running on port 8501"); break
    except Exception:
        time.sleep(1)
else:
    print("Streamlit did not start - see the log:")
    print(open("/content/streamlit.log").read()[-3000:])'''

CELL_PROXY = '''# STEP 7 (Option A, recommended) — Open the website through Colab's private port proxy
from google.colab.output import eval_js
url = eval_js("google.colab.kernel.proxyPort(8501, {'cache': false})")
print("Open this link in the SAME browser where you are signed in to Colab:")
print(url)'''

CELL_TUNNEL = '''# STEP 7 (Option B, fallback) — Public HTTPS link with a Cloudflare quick tunnel (no account needed)
# WARNING: anyone who has this random link can open the website while this Colab session runs.
import subprocess, re, time, os
CF = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
if not os.path.exists("/content/cloudflared"):
    !wget -q -O /content/cloudflared {CF}
    !chmod +x /content/cloudflared
tlog = open("/content/cloudflared.log", "w")
tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate",
                           "--url", "http://localhost:8501"], stdout=tlog, stderr=subprocess.STDOUT)
for _ in range(40):
    time.sleep(1)
    text = open("/content/cloudflared.log").read()
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", text)
    if m:
        print("Website URL:", m.group(0)); break
else:
    print("No URL yet - see /content/cloudflared.log")'''

CELL_STATUS = '''# STEP 8 — Status, logs and restart (run whenever something looks wrong)
import urllib.request
try:
    r = urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=3)
    print("Health:", r.read().decode())
except Exception as e:
    print("Streamlit is NOT running:", e, "-> run Step 6 again")
print("---- last lines of /content/streamlit.log ----")
print(open("/content/streamlit.log").read()[-2500:])'''

CELL_STOP = '''# STEP 9 — Stop the website (and the tunnel, if used)
!pkill -f "streamlit run" ; pkill -f cloudflared ; echo "stopped"'''

CELLS = [
    ("mount", "Check the runtime and mount Google Drive", CELL_MOUNT),
    ("paths", "Locate the website code and the trained models", CELL_PATHS),
    ("extract", "Get the website (GitHub clone or ZIP)", CELL_EXTRACT),
    ("install", "Install the requirements", CELL_INSTALL),
    ("models", "Provide the trained models and check them", CELL_MODELS),
    ("launch", "Launch the Streamlit app", CELL_LAUNCH),
    ("proxy", "Open the website — Option A: Colab proxy (private)", CELL_PROXY),
    ("tunnel", "Open the website — Option B: Cloudflare quick tunnel (public link)", CELL_TUNNEL),
    ("status", "Status, logs and restart", CELL_STATUS),
    ("stop", "Stop the website", CELL_STOP),
]
