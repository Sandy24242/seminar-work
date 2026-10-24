import { useEffect, useMemo, useState } from "react";
import { API_BASE, getHealth, getModelStatus, predict } from "./api.js";
import logo from "./logo.png";

const DISCLAIMER =
  "Experimental academic proof of concept. Results are model predictions from a photograph, not a professional " +
  "dental diagnosis. Please consult a qualified dentist.";
const ALLOWED = [".jpg", ".jpeg", ".png"];
const STAGE_ROWS = [
  ["stage1", "Stage 1 — Healthy vs Caries"],
  ["stage2", "Stage 2 — FDI tooth numbers"],
  ["localization", "Lesion localization"],
  ["stage3", "Stage 3 — G.V. Black class"],
  ["stage4", "Stage 4 — ICDAS score"],
];

const STREAMLIT_URL =
  import.meta.env.VITE_STREAMLIT_URL || "https://dental-caries-ai.streamlit.app";

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

function Badge({ status }) {
  return <span className={`badge badge-${String(status).toLowerCase()}`}>{status}</span>;
}

function LocalModelPanel({ health, status, error, checking }) {
  if (checking) {
    return <section className="card"><p className="muted">Checking local FastAPI backend at <code>{API_BASE}</code>…</p></section>;
  }

  if (error) {
    return (
      <section className="card">
        <h3>Local API Status: <Badge status="Offline" /></h3>
        <p className="muted small">
          No FastAPI server running at <code>{API_BASE}</code>.
        </p>
        <p className="small">
          To run predictions through this React interface, start the backend locally:
          <br />
          <code>cd backend && uvicorn api.main:app --host 127.0.0.1 --port 8000</code>
        </p>
        <p className="small" style={{ margin: 0 }}>
          <strong>For the live cloud demo with all 5 models preloaded:</strong> use the Streamlit Cloud application above.
        </p>
      </section>
    );
  }

  if (!status) return null;

  return (
    <section className="card">
      <h3>
        Local Model Status <Badge status={status.ready ? "Ready" : "Incomplete"} />
      </h3>
      <p className="muted small">
        API {health?.status === "ok" ? "healthy" : "unknown"} · TensorFlow {status.versions?.tensorflow || "n/a"} · Keras{" "}
        {status.versions?.keras || "n/a"}
      </p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Status</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {status.models.map((m) => (
            <tr key={m.key}>
              <td>
                {m.title}
                {!m.required && <span className="muted"> (optional)</span>}
                <div className="muted small">
                  <code>{m.file}</code>
                </div>
              </td>
              <td>
                <Badge status={m.loaded ? "Loaded" : "Unavailable"} />
              </td>
              <td className="small">{m.loaded ? (m.class_names || []).join(", ") || "loaded" : m.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!status.ready && (
        <p className="warn small">
          Required model(s) missing: {status.missing_required.join(", ")}. Stages that need them will report as unavailable.
        </p>
      )}
    </section>
  );
}

function validateFile(file, limits) {
  if (!file) return "Choose an image first.";
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  const allowed = limits?.allowed_extensions || ALLOWED;
  if (!allowed.includes(ext)) return `Unsupported file type "${ext}". Upload a JPG, JPEG or PNG image.`;
  const maxMb = limits?.max_upload_mb || 20;
  if (file.size > maxMb * 1024 * 1024) return `File is larger than ${maxMb} MB.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

function Results({ res, previewUrl }) {
  const s1 = res.stage1;
  return (
    <section className="card">
      <h2>Result</h2>
      <p className="overall">{res.overall}</p>
      {res.error && <p className="warn">{res.error}</p>}
      <table>
        <thead>
          <tr>
            <th>Stage</th>
            <th>Status</th>
            <th>Output</th>
          </tr>
        </thead>
        <tbody>
          {STAGE_ROWS.map(([k, title]) => {
            const st = res.stage_status?.[k];
            let out = res[k]?.reason || "";
            if (k === "stage1" && s1)
              out = `${s1.prediction}${s1.prediction === "UNCERTAIN" ? ` (leaning ${s1.leaning})` : ""} · confidence ${pct(
                s1.confidence,
              )} · P(caries) ${s1.p_caries.toFixed(3)} vs threshold ${s1.threshold.toFixed(3)}`;
            if (k === "stage2" && res.stage2?.status === "RAN") out = `${res.stage2.n_teeth} teeth detected`;
            if (k === "localization" && res.localization?.status === "RAN")
              out = `${res.localization.n_valid_crops} lesion(s) at score ≥ ${res.localization.score_threshold_used.toFixed(2)}`;
            if ((k === "stage3" || k === "stage4") && res[k]?.status === "RAN") out = `${res[k].n_lesions} lesion(s) classified`;
            return (
              <tr key={k}>
                <td>{title}</td>
                <td>{st && <Badge status={st} />}</td>
                <td className="small">{out}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="images">
        <figure>
          <img src={previewUrl} alt="Uploaded" />
          <figcaption>Uploaded image</figcaption>
        </figure>
        {res.annotated_jpeg_base64 && (
          <figure>
            <img src={`data:image/jpeg;base64,${res.annotated_jpeg_base64}`} alt="Annotated detections" />
            <figcaption>Model detections (green = lesion, orange = tooth)</figcaption>
          </figure>
        )}
      </div>

      {res.lesions?.length > 0 && (
        <>
          <h3>Lesions</h3>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Localizer score</th>
                <th>Tooth (FDI)</th>
                <th>G.V. Black</th>
                <th>ICDAS</th>
              </tr>
            </thead>
            <tbody>
              {res.lesions.map((l) => (
                <tr key={l.lesion}>
                  <td>L{l.lesion}</td>
                  <td>{l.localizer_score.toFixed(2)}</td>
                  <td>{l.fdi_tooth || <span className="muted">{l.fdi_status}</span>}</td>
                  <td>
                    {l.gvb_class ? `${l.gvb_display} (${pct(l.gvb_confidence)})` : <Badge status="Unavailable" />}
                    {l.gvb_uncertain && <span className="muted"> uncertain</span>}
                  </td>
                  <td>
                    {l.icdas != null ? `${l.icdas} (${pct(l.icdas_confidence)})` : <Badge status="Unavailable" />}
                    {l.icdas_uncertain && <span className="muted"> uncertain</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      <p className="disclaimer">{res.disclaimer || DISCLAIMER}</p>
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState(null);
  const [backendError, setBackendError] = useState(null);
  const [checking, setChecking] = useState(true);
  const [showDevMode, setShowDevMode] = useState(false);

  const [file, setFile] = useState(null);
  const [continueUncertain, setContinueUncertain] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  useEffect(() => {
    Promise.all([getHealth(), getModelStatus()])
      .then(([h, s]) => {
        setHealth(h);
        setStatus(s);
        setBackendError(null);
        setShowDevMode(true); // Automatically show dev panel if backend is running
      })
      .catch((e) => {
        setBackendError(e.message);
      })
      .finally(() => {
        setChecking(false);
      });
  }, []);

  const stage1Loaded = status?.models?.find((m) => m.key === "stage1")?.loaded;

  async function onSubmit(e) {
    e.preventDefault();
    setResult(null);
    const v = validateFile(file, status?.limits);
    if (v) return setError(v);
    setError(null);
    setBusy(true);
    try {
      setResult(await predict(file, continueUncertain));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header>
        <img src={logo} alt="" className="logo" />
        <div>
          <h1>Dental Caries AI</h1>
          <p className="muted">AI-assisted analysis of intraoral photographs — Multi-CNN Cascade System</p>
        </div>
      </header>
      <p className="disclaimer">{DISCLAIMER}</p>

      {/* Main Landing & Cloud Demo Card */}
      <section className="hero-card">
        <div className="hero-badge-row">
          <Badge status="Live" />
          <span className="muted small">Hosted on Streamlit Community Cloud (Zero Cost · 24/7 Available)</span>
        </div>
        <h2 className="hero-title">Interactive Cloud Demonstration</h2>
        <p className="hero-desc">
          The complete, verified four-stage deep learning pipeline (healthy screening, FDI tooth identification,
          lesion localization, G.V. Black classification, and ICDAS scoring) is deployed and running in the cloud.
          Includes an interactive 7-question oral-habits assessment and automated PDF diagnostic reports.
        </p>
        <a
          href={STREAMLIT_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="cta-btn"
        >
          Launch Streamlit Cloud Demo ↗
        </a>
        <div className="cta-note">
          Accessible from any browser or mobile device, even when your personal computer is turned off.
        </div>

        <div className="pipeline-grid">
          <div className="pipeline-box">
            <h4>Stage 1: Caries Screening</h4>
            <p>EfficientNetB0 binary classifier separates healthy intraoral photos from caries.</p>
          </div>
          <div className="pipeline-box">
            <h4>Stage 2: FDI Numbering</h4>
            <p>Custom tooth grid detector locates and maps individual teeth using FDI notation.</p>
          </div>
          <div className="pipeline-box">
            <h4>Stage 3: G.V. Black</h4>
            <p>Class-agnostic lesion localizer + ResNet50 assigns Class I through VI to each cavity.</p>
          </div>
          <div className="pipeline-box">
            <h4>Stage 4: ICDAS Score</h4>
            <p>EfficientNetB0 evaluates lesion severity on the ICDAS 1–6 dental grading scale.</p>
          </div>
        </div>
      </section>

      {/* Developer / Local API Mode */}
      <section className="dev-section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h3 style={{ margin: "0 0 4px" }}>Local Developer & API Mode</h3>
            <p className="muted small" style={{ margin: 0 }}>
              Use this section when testing the local FastAPI backend (<code>{API_BASE}</code>).
            </p>
          </div>
          <button
            type="button"
            className="dev-toggle"
            onClick={() => setShowDevMode((v) => !v)}
          >
            {showDevMode ? "Hide Local API" : "Show Local API"}
          </button>
        </div>

        {showDevMode && (
          <>
            <LocalModelPanel
              health={health}
              status={status}
              error={backendError}
              checking={checking}
            />

            {health?.status === "ok" && (
              <section className="card">
                <h2>Analyse an image (Local API)</h2>
                <form onSubmit={onSubmit}>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    onChange={(e) => {
                      setFile(e.target.files?.[0] || null);
                      setResult(null);
                      setError(null);
                    }}
                  />
                  {previewUrl && !result && <img className="preview" src={previewUrl} alt="Preview" />}
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={continueUncertain}
                      onChange={(e) => setContinueUncertain(e.target.checked)}
                    />
                    Continue the cascade when Stage 1 is uncertain
                  </label>
                  <button
                    type="submit"
                    disabled={busy || !file || backendError || !stage1Loaded}
                  >
                    {busy ? "Analysing…" : "Analyse"}
                  </button>
                  {status && !stage1Loaded && !backendError && (
                    <p className="warn">Analysis is disabled because the Stage 1 model is not loaded on the local backend.</p>
                  )}
                </form>
                {error && <p className="error-text">{error}</p>}
              </section>
            )}

            {result && <Results res={result} previewUrl={previewUrl} />}
          </>
        )}
      </section>
    </div>
  );
}
