"""
End-to-end browser check (optional; needs `pip install playwright` and a Chromium build).

Starts `streamlit run streamlit_app.py` against a model folder, then drives the real website in headless Chromium:
home -> Start Assessment -> patient form + 7 questions -> upload photo -> Analyze -> results -> download PDF.
Saves screenshots to the output folder.

    python tests/e2e_browser_check.py --models /path/to/models --image photo.jpg --out screenshots/
Without --models it builds the TEST stand-in models (not trained) in a temp folder and uses a synthetic photo.
"""
import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def shot(pg, out, name, text=None):
    """Scroll the section whose heading contains `text` to the top of the viewport, then screenshot."""
    if text:
        pg.get_by_text(text, exact=True).first.evaluate("e => e.scrollIntoView({block: 'start'})")
        pg.wait_for_timeout(700)
    pg.screenshot(path=os.path.join(out, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models"); ap.add_argument("--image"); ap.add_argument("--out", default=os.path.join(ROOT, "e2e_screenshots"))
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    if not a.models:
        from standin_models import make_synthetic_photo, to_png_bytes, write_standin_models
        a.models = write_standin_models(os.path.join(tmp, "models"))
        a.image = os.path.join(tmp, "synthetic_caries.png")
        open(a.image, "wb").write(to_png_bytes(make_synthetic_photo("caries")))
        print("Using TEST stand-in models (not trained) + synthetic photo")
    port = free_port()
    env = {**os.environ, "DCAI_MODELS_DIR": a.models}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "streamlit_app.py"), "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if urllib.request.urlopen(url + "/_stcore/health", timeout=2).read().decode().strip() == "ok":
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Streamlit did not start")
        print("Streamlit health check: ok on", url)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            kw = {"executable_path": a.chromium} if a.chromium else {}
            b = p.chromium.launch(**kw)
            pg = b.new_page(viewport={"width": 1366, "height": 900}, device_scale_factor=1)
            pg.goto(url, wait_until="networkidle")
            pg.get_by_text("Dental Caries AI Assessment System").first.wait_for(timeout=60000)
            pg.wait_for_timeout(1500)
            shot(pg, a.out, "01_home.png")
            shot(pg, a.out, "01b_home_stages.png", "The four-stage AI pipeline")
            pg.get_by_role("button", name="Start Assessment").click()
            pg.get_by_text("Oral-habits questionnaire").first.wait_for()
            pg.get_by_role("textbox", name="Patient name or ID (optional)").fill("Demo-01")
            pg.get_by_role("spinbutton", name="Age (years) *").fill("20")
            pg.get_by_role("spinbutton", name="Age (years) *").press("Tab")
            picks = ["No", "Once a day", "Sometimes", "Not sure", "6-12 months ago", "Yes", "Yes"]
            groups = pg.locator('[data-testid="stMain"] [data-testid="stRadio"]')
            assert groups.count() == 7, groups.count()
            for i, v in enumerate(picks):
                groups.nth(i).get_by_text(v, exact=True).click()
            pg.wait_for_timeout(300)
            shot(pg, a.out, "02_patient_form.png", "Patient information")
            shot(pg, a.out, "02b_questionnaire.png", "Q4. Do you use a fluoride toothpaste?")
            pg.get_by_role("button", name="Calculate risk score and continue").click()
            pg.get_by_text("9 / 13 points").wait_for(timeout=30000)
            print("Questionnaire result shown: 9 / 13 points, HIGH")
            pg.locator('input[type="file"]').set_input_files(a.image)
            pg.get_by_text("Uploaded image").first.wait_for(timeout=30000)
            pg.wait_for_timeout(1200)
            shot(pg, a.out, "03_risk_result.png", "Questionnaire result")
            shot(pg, a.out, "03b_upload.png", "Dental photograph")
            pg.get_by_role("button", name="Analyze Dental Image").click()
            pg.get_by_text("Assessment results").first.wait_for(timeout=300000)
            pg.wait_for_timeout(2500)
            shot(pg, a.out, "04_results_summary.png", "Assessment results")
            shot(pg, a.out, "04b_stage_cards.png", "Stage results")
            shot(pg, a.out, "04c_images.png", "Original and annotated image")
            shot(pg, a.out, "04d_lesions.png", "Per-lesion results")
            shot(pg, a.out, "04e_downloads.png", "Download the report")
            body = pg.inner_text("body")
            for s in ("Stage 1 — Healthy vs Caries", "Stage 2 — FDI tooth identification", "Cavity localization",
                      "Stage 3 — G.V. Black classification", "Stage 4 — ICDAS scoring", "Per-lesion results", "Download the report"):
                assert s in body, s
            with pg.expect_download() as dl:
                pg.get_by_role("button", name="PDF report").click()
            path = os.path.join(a.out, dl.value.suggested_filename); dl.value.save_as(path)
            assert open(path, "rb").read(5) == b"%PDF-", "downloaded file is not a PDF"
            print("PDF downloaded:", os.path.basename(path), os.path.getsize(path), "bytes")
            # mobile layout
            mp = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            mp.goto(url, wait_until="networkidle"); mp.get_by_text("Start Assessment").first.wait_for(timeout=60000)
            mp.wait_for_timeout(1500)
            mp.screenshot(path=os.path.join(a.out, "05_mobile_home.png"))
            b.close()
        print("E2E CHECK PASSED — screenshots in", a.out)
        return 0
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except Exception:
            out = ""
        if "Traceback" in (out or ""):
            print("---- streamlit log ----\n" + out[-4000:])


if __name__ == "__main__":
    sys.exit(main())
