"""
Builds docs/Dental_Caries_AI_Colab_Guide.pdf with ReportLab.
The code cells come from tools/colab_cells.py (the same text as colab/Run_Dental_Caries_AI_Website.ipynb), and the
project file list is read from the project folder, so the guide always matches the delivered ZIP.

    python tools/make_guide.py [output.pdf]
"""
import os
import sys
from datetime import date
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from colab_cells import (CELL_EXTRACT, CELL_INSTALL, CELL_LAUNCH, CELL_MODELS, CELL_MOUNT, CELL_PATHS,  # noqa: E402
                         CELL_PROXY, CELL_STATUS, CELL_STOP, CELL_TUNNEL)
import config  # noqa: E402

FD = os.path.join(ROOT, "assets", "fonts")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FD, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(FD, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(FD, "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("DVMB", os.path.join(FD, "DejaVuSansMono-Bold.ttf")))

NAVY, TEAL, TEAL2 = colors.HexColor("#0B2545"), colors.HexColor("#0E7C86"), colors.HexColor("#13A3B4")
LIGHT, SKY, MUTED = colors.HexColor("#EAF6F8"), colors.HexColor("#CDEBF0"), colors.HexColor("#5B6B7B")
W, H = A4
M = 17 * mm
CW = W - 2 * M

ST = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=27, leading=33, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=TEAL),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=17, leading=22, textColor=NAVY, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14, spaceAfter=4),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8.2, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.4, leading=11),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=8.4, leading=11, textColor=colors.white),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.6, leading=14, leftIndent=14, bulletIndent=3, spaceAfter=2),
    "num": ParagraphStyle("num", fontName="DV", fontSize=9.6, leading=14, leftIndent=18, bulletIndent=0, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.45, leading=9.6, textColor=colors.HexColor("#0F172A")),
    "out": ParagraphStyle("out", fontName="DVM", fontSize=7.3, leading=9.3, textColor=colors.HexColor("#14532D")),
    "caption": ParagraphStyle("caption", fontName="DV", fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=6),
    "toc1": ParagraphStyle("toc1", fontName="DVB", fontSize=10.5, leading=17, textColor=NAVY),
    "toc2": ParagraphStyle("toc2", fontName="DV", fontSize=9.3, leading=14, leftIndent=16, textColor=colors.HexColor("#334155")),
}


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=20 * mm, bottomMargin=17 * mm,
                         title="Dental Caries AI — Google Colab Guide", author="Dental Caries AI Assessment System",
                         subject="How to run the Dental Caries AI website in Google Colab")
        frame = Frame(M, 17 * mm, CW, H - 37 * mm, id="f")
        self.addPageTemplates([PageTemplate("cover", [Frame(M, 15 * mm, CW, H - 30 * mm, id="c")], onPage=self.cover_bg),
                               PageTemplate("body", [frame], onPage=self.decor)])

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and getattr(fl, "_toc", None):
            lvl, text = fl._toc
            key = f"k{id(fl)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=lvl, closed=False)
            self.notify("TOCEntry", (lvl, text, self.page, key))

    @staticmethod
    def cover_bg(c, d):
        c.saveState()
        c.setFillColor(colors.HexColor("#F3FAFC")); c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(TEAL); c.rect(0, H - 14 * mm, W, 14 * mm, stroke=0, fill=1)
        c.setFillColor(NAVY); c.rect(0, 0, W, 11 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white); c.setFont("DV", 8)
        c.drawString(M, 4.2 * mm, "Academic AI proof of concept — not a clinical diagnosis.")
        c.restoreState()

    @staticmethod
    def decor(c, d):
        c.saveState()
        c.setFillColor(TEAL); c.rect(0, H - 9 * mm, W, 9 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white); c.setFont("DVB", 8.5)
        c.drawString(M, H - 6 * mm, "Dental Caries AI Assessment System — Google Colab Guide")
        c.setFont("DV", 8); c.drawRightString(W - M, H - 6 * mm, f"v{config.APP_VERSION}")
        c.setStrokeColor(SKY); c.line(M, 12 * mm, W - M, 12 * mm)
        c.setFillColor(MUTED); c.setFont("DV", 8)
        c.drawString(M, 7.5 * mm, "Dental_Caries_AI_Colab_Guide.pdf")
        c.drawRightString(W - M, 7.5 * mm, f"Page {d.page}")
        c.restoreState()


def P(text, style="body"):
    return Paragraph(text, ST[style])


def H1(text):
    p = Paragraph(escape(text), ST["h1"]); p._toc = (0, text); return p


def H2(text, toc=True):
    p = Paragraph(escape(text), ST["h2"])
    if toc:
        p._toc = (1, text)
    return p


def bullets(items, style="bullet"):
    return [Paragraph(t, ST[style], bulletText="•") for t in items]


def numbered(items):
    return [Paragraph(t, ST["num"], bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def box(flowables, bg, border, pad=7):
    t = Table([[flowables]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, border),
                           ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                           ("TOPPADDING", (0, 0), (-1, -1), pad - 1), ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 1)]))
    return t


def code(src, label="Code cell — copy into Colab"):
    lab = Paragraph(f'<font name="DVB" color="#0E7C86" size="7.5">{escape(label)}</font>', ST["small"])
    return box([lab, Spacer(1, 2), Preformatted(src, ST["code"])], colors.HexColor("#F1F5F9"), colors.HexColor("#CBD5E1"))


def expected(src, label="Expected output (values such as sizes and versions may differ slightly)"):
    lab = Paragraph(f'<font name="DVB" color="#15803D" size="7.5">{escape(label)}</font>', ST["small"])
    return box([lab, Spacer(1, 2), Preformatted(src, ST["out"])], colors.HexColor("#F0FDF4"), colors.HexColor("#BBF7D0"))


def note(text, kind="tip"):
    c = {"tip": ("#EFF8FF", "#B2DDFF", "Tip"), "warn": ("#FFF8EC", "#F5D9A8", "Important"),
         "sec": ("#FDF2F2", "#F5C2C2", "Security")}[kind]
    return box([P(f'<font name="DVB">{c[2]}:</font> {text}', "body")], colors.HexColor(c[0]), colors.HexColor(c[1]))


def img(path, width, caption=None, keep=True):
    if not os.path.isfile(path):
        return [P(f"[image missing: {escape(os.path.basename(path))}]", "small")]
    from PIL import Image as PI
    w, h = PI.open(path).size
    im = Image(path, width=width, height=width * h / w)
    out = [im]
    if caption:
        out.append(P(escape(caption), "caption"))
    return [KeepTogether(out)] if keep else out


def table(rows, widths, head=True, font=8.4):
    data = [[Paragraph(str(c) if isinstance(c, str) else c, ST["cellb" if (head and i == 0) else "cell"]) if isinstance(c, str) else c
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), TEAL))
    for i in range(1 if head else 0, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F4FAFB")))
    t.setStyle(TableStyle(st))
    return t


# ---------------------------------------------------------------------------------------------- diagrams
def arch_diagram():
    d = Drawing(CW, 190)
    def blk(x, y, w, h, title, lines, fill):
        d.add(Rect(x, y, w, h, rx=8, ry=8, fillColor=fill, strokeColor=TEAL, strokeWidth=1))
        d.add(String(x + 8, y + h - 15, title, fontName="DVB", fontSize=9, fillColor=NAVY))
        for i, l in enumerate(lines):
            d.add(String(x + 8, y + h - 30 - i * 11.5, l, fontName="DV", fontSize=7.4, fillColor=colors.HexColor("#334155")))
    def arrow(x1, y1, x2, y2, label=None):
        d.add(Line(x1, y1, x2, y2, strokeColor=TEAL2, strokeWidth=2))
        import math
        a = math.atan2(y2 - y1, x2 - x1); s = 6
        d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.45), y2 - s * math.sin(a - 0.45),
                       x2 - s * math.cos(a + 0.45), y2 - s * math.sin(a + 0.45)], fillColor=TEAL2, strokeColor=TEAL2))
        if label:
            d.add(String((x1 + x2) / 2 - 22, (y1 + y2) / 2 + 4, label, fontName="DV", fontSize=6.8, fillColor=MUTED))
    blk(0, 98, 150, 88, "Google Drive", ["MyDrive/dentalcaries/", " ├ Dental_Caries_AI_Website.zip", " └ pipeline_stage1_to_4_corrected/",
                                         "     5 × .keras + 5 × metadata"], colors.HexColor("#FFFFFF"))
    d.add(Rect(170, 0, 190, 186, rx=10, ry=10, fillColor=LIGHT, strokeColor=NAVY, strokeWidth=1))
    d.add(String(180, 172, "Google Colab virtual machine", fontName="DVB", fontSize=9, fillColor=NAVY))
    blk(180, 98, 170, 64, "/content/Dental_Caries_AI_Website", ["app.py, pipeline/, assets/ …", "models/  (copied from Drive)"], colors.white)
    blk(180, 12, 170, 70, "Streamlit server :8501", ["loads the 5 trained models once", "runs Stage 1 → 4 per upload",
                                                     "builds PDF / TXT / JSON / CSV"], colors.white)
    blk(380, 98, CW - 380, 88, "Your browser", ["Option A: Colab proxy link", "  (private to your Google login)",
                                                "Option B: trycloudflare.com link", "  (public random URL)"], colors.white)
    blk(380, 12, CW - 380, 70, "You", ["questionnaire + photo upload", "results dashboard", "report downloads"], colors.white)
    arrow(150, 150, 180, 140, "unzip")
    arrow(150, 112, 180, 108, "copy")
    arrow(265, 98, 265, 82)
    arrow(350, 47, 380, 130, "proxy / tunnel")
    arrow(380 + (CW - 380) / 2, 98, 380 + (CW - 380) / 2, 82)
    return d


def drive_diagram():
    d = Drawing(CW, 120)
    d.add(Rect(0, 0, CW, 120, rx=8, ry=8, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
    d.add(Rect(0, 100, CW, 20, rx=8, ry=8, fillColor=colors.HexColor("#F1F5F9"), strokeColor=colors.HexColor("#CBD5E1")))
    d.add(String(10, 106, "drive.google.com  —  Shared with me", fontName="DVB", fontSize=8.5, fillColor=NAVY))
    d.add(Rect(10, 62, 150, 26, rx=4, ry=4, fillColor=SKY, strokeColor=TEAL))
    d.add(String(18, 71, "dentalcaries  (folder)", fontName="DVB", fontSize=8.5, fillColor=NAVY))
    d.add(Rect(170, 10, 175, 82, rx=5, ry=5, fillColor=colors.white, strokeColor=colors.HexColor("#94A3B8")))
    for i, t in enumerate(["Open with", "Share", "Organize  ▸", "Download"]):
        y = 74 - i * 18
        if i == 2:
            d.add(Rect(172, y - 5, 171, 17, fillColor=LIGHT, strokeColor=None))
        d.add(String(182, y, t, fontName="DVB" if i == 2 else "DV", fontSize=8.2, fillColor=NAVY))
    d.add(Rect(352, 30, 150, 44, rx=5, ry=5, fillColor=colors.white, strokeColor=colors.HexColor("#94A3B8")))
    d.add(Rect(354, 50, 146, 18, fillColor=LIGHT, strokeColor=None))
    d.add(String(362, 55, "Add shortcut", fontName="DVB", fontSize=8.2, fillColor=NAVY))
    d.add(String(362, 37, "Move", fontName="DV", fontSize=8.2, fillColor=MUTED))
    d.add(String(10, 40, "1. right-click the folder", fontName="DV", fontSize=7.6, fillColor=MUTED))
    d.add(String(10, 28, "2. Organize ▸ Add shortcut", fontName="DV", fontSize=7.6, fillColor=MUTED))
    d.add(String(10, 16, "3. choose “My Drive” ▸ Add", fontName="DV", fontSize=7.6, fillColor=MUTED))
    return d


def runtime_diagram():
    d = Drawing(CW, 104)
    d.add(Rect(0, 0, CW, 104, rx=8, ry=8, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
    menu = ["File", "Edit", "View", "Insert", "Runtime", "Tools", "Help"]
    x = 10
    for m in menu:
        if m == "Runtime":
            d.add(Rect(x - 4, 82, 52, 16, fillColor=LIGHT, strokeColor=None))
        d.add(String(x, 86, m, fontName="DVB" if m == "Runtime" else "DV", fontSize=8.5, fillColor=NAVY))
        x += 50
    d.add(Rect(196, 8, 170, 72, rx=4, ry=4, fillColor=colors.white, strokeColor=colors.HexColor("#94A3B8")))
    for i, t in enumerate(["Run all", "Restart session", "Change runtime type", "Disconnect and delete runtime"]):
        y = 64 - i * 16
        if i == 2:
            d.add(Rect(198, y - 4, 166, 15, fillColor=LIGHT, strokeColor=None))
        d.add(String(206, y, t, fontName="DVB" if i == 2 else "DV", fontSize=8, fillColor=NAVY))
    d.add(Rect(380, 8, CW - 390, 72, rx=4, ry=4, fillColor=colors.white, strokeColor=TEAL))
    d.add(String(390, 64, "Hardware accelerator", fontName="DVB", fontSize=8, fillColor=NAVY))
    for i, t in enumerate(["( ) CPU", "(•) T4 GPU", "( ) other GPUs / TPU"]):
        d.add(String(392, 48 - i * 14, t, fontName="DVB" if i == 1 else "DV", fontSize=8, fillColor=TEAL if i == 1 else MUTED))
    return d


# ---------------------------------------------------------------------------------------------- content
def project_tree(width=92):
    """Top-level files and folders of the delivered project, wrapped to fit the page."""
    import textwrap
    skip = {"__pycache__", ".pytest_cache", "e2e_screenshots"}
    lines = []
    for top in sorted(os.listdir(ROOT)):
        if top in skip or top.endswith(".pyc"):
            continue
        p = os.path.join(ROOT, top)
        if os.path.isdir(p):
            files = sorted(f for f in os.listdir(p) if f not in skip)
            lines += textwrap.wrap(", ".join(files), width, initial_indent=f"{top + '/':12}",
                                   subsequent_indent=" " * 12, break_on_hyphens=False) or [top + "/"]
        else:
            lines.append(top)
    return "\n".join(lines)


def build(out):
    sh = lambda n: os.path.join(ROOT, "docs", "screenshots", n)
    story = []
    # ---------------- cover
    story += [Spacer(1, 6 * mm)]
    logo = os.path.join(ROOT, "assets", "logo.png")
    story.append(Table([[Image(logo, 24 * mm, 24 * mm), [P("Dental Caries AI Assessment System", "title")]]],
                       colWidths=[30 * mm, CW - 30 * mm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [Spacer(1, 4 * mm), P("Step-by-step guide: running the website in Google Colab", "subtitle"), Spacer(1, 2 * mm),
              P(f"For <b>Dental_Caries_AI_Website.zip</b> (version {config.APP_VERSION}) · entry point <b>app.py</b> · "
                f"Colab notebook <b>colab/Run_Dental_Caries_AI_Website.ipynb</b> · {date.today():%d %B %Y}", "body"),
              Spacer(1, 5 * mm)]
    story += img(os.path.join(ROOT, "assets", "hero_dental_ai.jpg"), CW)
    story += [Spacer(1, 4 * mm), box([P("<b>What this guide covers</b>"),
                                      *bullets(["opening Google Colab with the right Google account and reaching the project in Google Drive",
                                                "uploading and extracting the website ZIP, installing the requirements",
                                                "giving the website the real trained models from <i>pipeline_stage1_to_4_corrected</i>",
                                                "launching Streamlit and opening the website link (private Colab proxy or public tunnel)",
                                                "running a complete assessment, downloading the report, restarting after a disconnect",
                                                "troubleshooting and a final checklist"])], colors.white, SKY, 9)]
    story += [Spacer(1, 4 * mm), P(escape(config.ACADEMIC_DISCLAIMER), "small"), NextPageTemplate("body"), PageBreak()]

    # ---------------- TOC
    toc = TableOfContents(); toc.levelStyles = [ST["toc1"], ST["toc2"]]; toc.dotsMinLevel = 0
    story += [P("Contents", "h1"), toc, PageBreak()]

    # ---------------- 1 overview
    story += [H1("1. Overview — what you are going to run")]
    story += [P("The website wraps the four-stage pipeline you trained in <b>Dental_Caries_Pipeline_Stage1_to_Stage4_with_FDI_Stage2.ipynb</b>. "
                "It does <b>not</b> retrain anything: it loads the saved <b>.keras</b> models and their <b>*_metadata.json</b> files and reuses "
                "the notebook's preprocessing, thresholds, class mappings, lesion→tooth association and report logic.")]
    story += img(os.path.join(ROOT, "assets", "workflow_stages.png"), CW, "Figure 1 — Assessment workflow: questionnaire, Stage 1 to Stage 4, report.")
    story += [H2("How the pieces fit together in Colab"), arch_diagram(),
              P("Figure 2 — The ZIP and the models stay in Google Drive. Each Colab session extracts the website to the local disk, "
                "copies the models next to it and starts a Streamlit server, which you open through a link.", "caption")]
    story += [H2("The trained models used by the website")]
    story.append(table([["Stage", "File in pipeline_stage1_to_4_corrected/", "Size", "Settings read from metadata"],
                        ["Stage 1", "stage1_caries_model.keras", "45.5 MB", "EfficientNetB0 · 224 px · threshold 0.603 ± 0.07 · confidence ≥ 0.60"],
                        ["Stage 2", "stage2_fdi_detector.keras", "29.5 MB", "26 FDI classes · tooth score ≥ 0.55 · number confidence ≥ 0.50"],
                        ["Localizer", "cavity_localizer_model.keras", "29.0 MB", "input 416 px · lesion score ≥ 0.40"],
                        ["Stage 3", "stage3_gvblack_model.keras", "272.0 MB", "ResNet50 · G.V. Black Class I–VI · context pad 0.5"],
                        ["Stage 4", "stage4_icdas_model.keras", "50.4 MB", "EfficientNetB0 · ICDAS 1–6 (no ICDAS 0) · window ×1.5"]],
                       [17 * mm, 52 * mm, 17 * mm, CW - 86 * mm]))
    story += [Spacer(1, 3), P("Values as printed by Section 75 of your notebook and stored in the metadata files on Drive. Each .keras file "
                              "needs its matching <b>*_metadata.json</b> (same name + _metadata).", "small")]
    story.append(note("The models are <b>not inside the ZIP</b> (≈ 426 MB in total). The website reads them from your Drive folder. "
                      "If a model is missing, its stage is shown as <b>Unavailable</b> — nothing is ever invented or replaced.", "warn"))

    # ---------------- 2 prerequisites
    story += [PageBreak(), H1("2. Prerequisites")]
    story += bullets(["A Google account that can open the Drive folder <b>dentalcaries</b> "
                      "(owner: your PCCOE account; it is shared with your Gmail account).",
                      "The folder <b>dentalcaries/pipeline_stage1_to_4_corrected</b> with the 5 model files and 5 metadata files (Section 1).",
                      "The file <b>Dental_Caries_AI_Website.zip</b> (this deliverable) on your computer.",
                      "A desktop browser (Chrome, Edge or Firefox). Colab works best on a computer; the website itself also works on a phone.",
                      "About 15 minutes for the first run; about 5 minutes for later runs.",
                      "No API keys, tokens or paid services are needed."])
    story += [H2("Files inside the ZIP (generated from the delivered project)")]
    story.append(code(project_tree(), "Dental_Caries_AI_Website/"))
    story.append(note("Keep the ZIP in Drive. Do not extract it inside Drive — Colab extracts it to its own fast disk in Step 3.", "tip"))

    # ---------------- 3 Colab + account
    story += [PageBreak(), H1("3. Open Google Colab with the right account")]
    story += numbered(["Go to <b>https://colab.research.google.com</b>.",
                       "Look at the round profile picture in the top-right corner. Click it and choose the account that can see "
                       "<b>dentalcaries</b> (the PCCOE account that owns it, or the Gmail account it is shared with). "
                       "Colab and Drive must use the <b>same</b> account — Drive is mounted as whoever is signed in to Colab.",
                       "Open the setup notebook: <b>File → Upload notebook</b> → choose "
                       "<b>colab/Run_Dental_Caries_AI_Website.ipynb</b> from the extracted ZIP on your computer. "
                       "(Alternative: <b>File → New notebook</b> and paste the code cells from this guide one by one.)",
                       "Choose a GPU: <b>Runtime → Change runtime type → T4 GPU → Save</b>. A CPU runtime also works, but loading "
                       "the models and analysing a photo are slower."])
    story += [runtime_diagram(), P("Figure 3 — Runtime → Change runtime type → T4 GPU (schematic).", "caption")]

    story += [H1("4. Access the project in Google Drive")]
    story += [P("The notebook expects the project at <b>MyDrive/dentalcaries</b>, exactly like your training notebook "
                "(<i>DRIVE_BASE = \"/content/drive/MyDrive/dentalcaries\"</i>).")]
    story += [H2("If you use the account that OWNS the folder"), P("Nothing to do — it is already at My Drive → dentalcaries.")]
    story += [H2("If the folder is SHARED with you (e.g. your Gmail account)"),
              P("Shared folders are not inside “My Drive”, so Colab cannot see them until you add a shortcut:")]
    story += numbered(["Open <b>https://drive.google.com</b> with the same account as Colab.",
                       "Click <b>Shared with me</b>, right-click the folder <b>dentalcaries</b>.",
                       "Choose <b>Organize → Add shortcut</b>, select <b>My Drive</b>, click <b>Add</b>.",
                       "Check: <b>My Drive</b> now shows <b>dentalcaries</b> with a small arrow icon. "
                       "Colab sees it as <b>/content/drive/MyDrive/dentalcaries</b>."])
    story += [drive_diagram(), P("Figure 4 — Adding a shortcut to a shared folder (schematic).", "caption")]
    story.append(note("If you only have <b>view</b> access to the shared folder you cannot upload the ZIP into it. Upload the ZIP to "
                      "<b>My Drive</b> (root) instead — the notebook also looks there.", "tip"))
    story += [H2("Upload the website ZIP to Drive")]
    story += numbered(["In Drive open <b>My Drive → dentalcaries</b>.",
                       "Click <b>+ New → File upload</b> and choose <b>Dental_Caries_AI_Website.zip</b>.",
                       "Wait until the upload panel says “1 upload complete”. The file name must stay exactly "
                       "<b>Dental_Caries_AI_Website.zip</b> (rename it if the browser added “(1)”)."])

    # ---------------- 5 notebook steps
    story += [PageBreak(), H1("5. Run the setup notebook, step by step")]
    story += [P("Run each cell with the ▶ button (or Shift+Enter) and wait until it finishes before starting the next one. "
                "The cells below are identical to the ones in <b>colab/Run_Dental_Caries_AI_Website.ipynb</b>.")]

    story += [H2("Step 1 — Check the runtime and mount Google Drive"), code(CELL_MOUNT),
              P("A pop-up asks for permission to access Google Drive: choose the <b>same account</b> as in Section 3 and allow access."),
              expected("GPU 0: Tesla T4 (UUID: GPU-…)\nMounted at /content/drive\nDrive mounted: True")]
    story += [H2("Step 2 — Locate the website ZIP and the trained models"), code(CELL_PATHS),
              expected("Website ZIP : /content/drive/MyDrive/dentalcaries/Dental_Caries_AI_Website.zip\n"
                       "Model folder: /content/drive/MyDrive/dentalcaries/pipeline_stage1_to_4_corrected -> found\n"
                       "   cavity_localizer_model.keras          29.0 MB\n   stage1_caries_model.keras             45.5 MB\n"
                       "   stage2_fdi_detector.keras             29.5 MB\n   stage3_gvblack_model.keras           272.0 MB\n"
                       "   stage4_icdas_model.keras              50.4 MB")]
    story.append(note("If you see <b>NOT FOUND</b>, fix the Drive location (Section 4) and run Step 2 again. If your models live in a "
                      "different folder, change <b>MODELS_DRIVE</b> in this cell.", "warn"))
    story += [CondPageBreak(90 * mm), H2("Step 3 — Extract the website"), code(CELL_EXTRACT),
              expected("OK      app.py\nOK      config.py\nOK      requirements-colab.txt\nOK      pipeline/runner.py\n"
                       "OK      assets/logo.png\n/content/Dental_Caries_AI_Website")]
    story += [H2("Step 4 — Install the requirements"), code(CELL_INSTALL),
              P("Only <b>streamlit</b> and <b>reportlab</b> are installed. TensorFlow, Keras, NumPy, pandas, OpenCV, Pillow and Plotly "
                "come with Colab. Do <b>not</b> reinstall TensorFlow: your models were trained and saved with Colab's own "
                "TensorFlow 2.21.0 / Keras 3.13.2."),
              expected("TensorFlow 2.21.0 | Keras 3.13.2\nStreamlit 1.x.x | ReportLab 4.x")]
    story += [CondPageBreak(90 * mm), H2("Step 5 — Provide the trained models and check them"), code(CELL_MODELS),
              P("The ten files are copied from Drive to <b>/content/Dental_Caries_AI_Website/models</b> (about 1–3 minutes the first "
                "time; skipped later in the same session). <b>scripts/check_models.py</b> then loads every model exactly like the website does."),
              expected("Model folder: /content/Dental_Caries_AI_Website/models\n  OK   stage1_caries_model.keras  (45.5 MB)\n"
                       "  OK   stage1_caries_model_metadata.json  (0.0 MB)\n  … (all 10 files OK)\n\n"
                       "Loading models (this can take 1–2 minutes)…\n\nTensorFlow 2.21.0 | Keras 3.13.2\n"
                       "  [Loaded     ] Stage 1 — Healthy vs Caries (CNN 1)\n"
                       "                EfficientNetB0 · 224 px · threshold 0.603 ± 0.07 · confidence ≥ 0.60\n"
                       "  [Loaded     ] Stage 2 — FDI tooth detector\n"
                       "                26 FDI classes · score ≥ 0.55 · number confidence ≥ 0.50\n"
                       "  [Loaded     ] Cavity / lesion localizer      input 416 px · score ≥ 0.40\n"
                       "  [Loaded     ] Stage 3 — G.V. Black classifier (CNN 2)   ResNet50 · … · pad 0.5\n"
                       "  [Loaded     ] Stage 4 — ICDAS classifier (CNN 3)   EfficientNetB0 · … · scores [1, … 6]\n\n"
                       "RESULT: all required models loaded\n"
                       "(the real script prints each model on one line — the layout above is wrapped to fit the page)")]
    story += [CondPageBreak(110 * mm), H2("Step 6 — Launch the Streamlit app"), code(CELL_LAUNCH),
              P("Streamlit runs in the background, so the notebook stays usable. The server listens on port <b>8501</b>; its log is "
                "written to <b>/content/streamlit.log</b>. The flags <i>enableCORS false</i> and <i>enableXsrfProtection false</i> are "
                "needed because the page is reached through a proxy or tunnel (otherwise photo uploads can fail with error 403)."),
              expected("Streamlit is running on port 8501")]
    story.append(note("Streamlit does <b>not</b> become reachable from the internet by itself — Colab machines have no public address. "
                      "Use Option A or Option B in Step 7 to open it.", "warn"))
    story += [CondPageBreak(80 * mm), H2("Step 7 — Open the website (choose ONE option)")]
    story += [P("<b>Option A — Colab's built-in proxy (recommended, private).</b> Google serves port 8501 through a link that only works "
                "for your signed-in Google account in this browser. No extra software, no account, nothing public."), code(CELL_PROXY),
              expected("Open this link in the SAME browser where you are signed in to Colab:\nhttps://…-8501-…colab… (a long Google address)")]
    story += [CondPageBreak(100 * mm),
              P("<b>Option B — Cloudflare quick tunnel (fallback, public link).</b> Use it if the Option A page stays blank or keeps "
                "reconnecting, or when somebody else must open the website (e.g. an examiner on another computer). It downloads "
                "Cloudflare's open-source <i>cloudflared</i> program and creates a temporary random <i>trycloudflare.com</i> address. "
                "No account or token is needed and none is stored in the notebook."), code(CELL_TUNNEL),
              expected("Website URL: https://random-words-1234.trycloudflare.com")]
    story.append(note("A tunnel URL is <b>public</b>: anyone who knows it can use the website (and upload photos to your Colab session) "
                      "while the session runs. Share it only with people who need it, never post it, and stop it with Step 9 when you "
                      "are finished. The link changes every time and stops working when the session ends. Quick tunnels have no uptime "
                      "guarantee and are meant for testing/demos. (Alternative: <b>ngrok</b> with your own free auth token stored in "
                      "Colab's <i>Secrets</i> panel — never paste a token into a shared notebook.)", "sec"))
    story += [CondPageBreak(70 * mm), H2("Step 8 — Status, logs and restart"), code(CELL_STATUS),
              expected("Health: ok\n---- last lines of /content/streamlit.log ----\n  You can now view your Streamlit app in your browser.\n\n  Local URL: http://localhost:8501\n  Network URL: http://172.28.0.12:8501")]
    story += [H2("Step 9 — Stop the website"), code(CELL_STOP), expected("stopped")]

    # ---------------- 6 using the website
    story += [PageBreak(), H1("6. Run a sample assessment")]
    story.append(note("The screenshots in this section were taken from the real website during automated testing. Because the trained "
                      "weights could not be downloaded into the test environment, they were captured with <b>test-only stand-in "
                      "models</b> on a <b>synthetic test image</b>. The layout is exactly what you will see; the predictions shown "
                      "here are <b>not</b> real. Section 7 shows the real reference result from your notebook.", "warn"))
    steps = [("Home page", "01_home.png", "The home page with the hero image and the Start Assessment button. Click "
              "<b>Start Assessment</b> (or <b>Assessment</b> in the left menu)."),
             ("Patient details", "02_patient_form.png", "Enter an optional name/ID and the <b>age</b> (required, 1–120). The date "
              "and time are recorded automatically when you submit."),
             ("Questionnaire", "02b_questionnaire.png", "Answer all seven questions, then click <b>Calculate risk score and "
              "continue</b>. Missing answers or an invalid age are reported in red."),
             ("Questionnaire result", "03_risk_result.png", "Score out of 13, risk category (LOW 0–3, MODERATE 4–7, HIGH 8–13), points per "
              "answer and personalised advice. Click <b>Edit answers</b> to change them."),
             ("Upload the photo", "03b_upload.png", "Drag a JPG/JPEG/PNG intraoral photo onto the upload box. The preview shows the "
              "file name, format and dimensions; <b>Remove image</b> clears it, dropping another file replaces it. Then click "
              "<b>Analyze Dental Image</b>. A progress list shows each pipeline step as it actually completes."),
             ("Results summary", "04_results_summary.png", "Patient, questionnaire risk and the overall pipeline result."),
             ("Stage cards", "04b_stage_cards.png", "One card per stage with its output, confidence and a status: Completed, "
              "Uncertain, Unavailable, Skipped or Failed."),
             ("Images", "04c_images.png", "Original upload next to the annotated image: orange = teeth with FDI number ('?' = "
              "uncertain), green = lesions labelled L&lt;id&gt; T&lt;tooth&gt;: G.V. Black | ICDAS. Tabs show each image large (useful on phones)."),
             ("Per-lesion table", "04d_lesions.png", "Lesion ID, FDI tooth, tooth confidences, localization confidence, G.V. Black and "
              "ICDAS with confidences and uncertainty flags. Expand a lesion to see its model input crops and probability charts."),
             ("Downloads", "04e_downloads.png", "Recommendations and the download buttons (Section 8)."),
             ]
    for i, (t, f, d) in enumerate(steps, 1):
        story.append(KeepTogether([H2(f"6.{i} {t}", toc=False), P(d), *img(sh(f), CW * 0.66, keep=False), Spacer(1, 6)]))
    story.append(KeepTogether([H2("On a phone", toc=False), P("The layout adapts to small screens; the menu moves behind the » button."),
                               *img(sh("05_mobile_home.png"), 55 * mm, "Home page at phone width.", keep=False)]))

    # ---------------- 7 complete example
    story += [PageBreak(), H1("7. Complete start-to-finish example (reference result)")]
    story += [P("This example repeats the assessment that your notebook ran in Section 82.1 (output saved in the notebook), so you "
                "can check that the website gives the same answer with the same models and the same photo.")]
    story += numbered(["Find the photo <b>UFV72.jpg</b> (the photo used in the notebook's Section 82.1 example; it comes from your "
                       "project datasets — search for it inside <b>caries_FDI_dataset_final.zip</b> on Drive) and download it to your computer.",
                       "Run Steps 1–7 of Section 5 and open the website.",
                       "Assessment: ID <b>20</b>, age <b>20</b>; answers Q1 <b>No</b>, Q2 <b>Once a day</b>, Q3 <b>Sometimes</b>, "
                       "Q4 <b>Not sure</b>, Q5 <b>6-12 months ago</b>, Q6 <b>Yes</b>, Q7 <b>Yes</b>.",
                       "Expected questionnaire result: <b>HIGH — 9 / 13</b> (points 2, 1, 1, 1, 1, 2, 1).",
                       "Upload <b>UFV72.jpg</b> and click <b>Analyze Dental Image</b> (default options).",
                       "Compare with the notebook's recorded result below."])
    story.append(table([["Stage", "Notebook result for UFV72.jpg (recorded 19 Oct 2026)"],
                        ["Stage 1", "CARIES — confidence 100.0% (P(caries) 1.000, threshold 0.603)"],
                        ["Stage 2", "3 teeth: 11 (number 100.0%, detection 82.5%), Unknown / uncertain — best guess 11 (83.4%, 69.6%), 21 (100.0%, 73.6%)"],
                        ["Localization", "2 regions at score ≥ 0.40 (validated 0.40); 2 valid lesion crops"],
                        ["Lesion 1", "tooth 21 · localization 65.6% · G.V. Black Class IV 82.8% · ICDAS 5 60.0%"],
                        ["Lesion 2", "tooth Unknown / uncertain (best guess 11) · localization 53.4% · Class IV 100.0% · ICDAS 5 57.9%"]],
                       [26 * mm, CW - 26 * mm]))
    story += [Spacer(1, 4)]
    story += img(os.path.join(ROOT, "docs", "images", "notebook_reference_annotated_UFV72.png"), CW * 0.92,
                 "Figure 5 — Real annotated output of your notebook for UFV72.jpg (orange = teeth/FDI, green = lesions). The website "
                 "draws the same boxes and labels with the same code.")
    story.append(note("The website's pipeline was tested to produce <b>identical</b> results to the notebook's own functions "
                      "(Sections 6.1, 6.6, 6.7, 76, 82) on the same models and images. Small differences can only come from a "
                      "different file (e.g. a re-saved or resized copy of the photo), a different TensorFlow version or GPU vs CPU "
                      "rounding (normally < 0.1 %).", "tip"))

    # ---------------- 8 downloads
    story += [H1("8. Download the final report")]
    story.append(table([["Button", "File", "Contents"],
                        ["PDF report", "dental_assessment_<time>_<id>_report.pdf", "Professional report: summary, questionnaire, stage table, images, teeth, lesions, advice, limitations, model info"],
                        ["TXT summary", "…_summary.txt", "The notebook's FINAL DENTAL REPORT text (sections 1–10)"],
                        ["JSON results", "…_results.json", "All structured results, stage statuses, pipeline steps, model information"],
                        ["CSV per lesion", "…_lesions.csv", "One row per lesion with confidences and uncertainty flags"],
                        ["Annotated image", "…_annotated.jpg", "The photo with tooth and lesion boxes"],
                        ["Everything (ZIP)", "…_all_results.zip", "All five files above"]],
                       [30 * mm, 58 * mm, CW - 88 * mm]))
    story += [Spacer(1, 3), P("Files download straight to your computer through the browser (the Downloads folder). They are generated in "
                              "memory and are not saved in Colab or Drive. To keep a copy in Drive, upload the downloaded files to a Drive folder.")]

    # ---------------- 9 restart / preserve
    story += [H1("9. After a disconnect — restart and keep your work")]
    story += bullets(["Colab sessions end after inactivity or after some hours; the local disk (/content) is then wiped. "
                      "<b>Your ZIP and models are safe in Drive</b>; only the running website stops.",
                      "To restart: <b>Runtime → Run all</b> is fine, but you only need Steps 1–7 (choose one Option in Step 7). "
                      "Then open the <b>new</b> link — old links stop working.",
                      "If only the website stopped (Step 8 says “NOT running”) while the session is alive: run Step 6 and Step 7 again.",
                      "Keep the setup notebook in Drive: <b>File → Save a copy in Drive</b> (it is stored in <i>My Drive/Colab Notebooks</i>).",
                      "To update the website, replace <b>Dental_Caries_AI_Website.zip</b> in Drive and run Steps 3–7 again.",
                      "Download reports before closing the browser tab — an assessment lives only in that browser session."])

    # ---------------- 10 troubleshooting
    story += [PageBreak(), H1("10. Troubleshooting")]
    rows = [["Problem", "What you see", "Fix"],
            ["Drive permission", "Step 1 pop-up blocked, or <i>MyDrive</i> missing", "Allow pop-ups for colab.research.google.com, run Step 1 again and accept with the same account as Colab. If needed: <i>drive.flush_and_unmount()</i>, then Step 1."],
            ["Wrong account / shared folder", "Step 2: model folder NOT FOUND", "Check the account (Section 3). For a shared folder add the shortcut to My Drive (Section 4), then rerun Step 2."],
            ["Missing files", "Step 2 lists fewer than 5 .keras files, or Step 5 shows MISSING", "Open dentalcaries/pipeline_stage1_to_4_corrected in Drive and check the 10 file names in Section 1. Names must not change (no “(1)”). Stage 2 (FDI) is optional; the others are required."],
            ["ZIP not found / incomplete extraction", "Step 2: ZIP NOT FOUND; Step 3: “ZIP is damaged” or MISSING lines", "Wait for the Drive upload to finish, check the exact file name, upload again if damaged, rerun Steps 2–3."],
            ["Dependency conflict", "pip warnings in Step 4", "Warnings about other Colab packages are harmless. Never install another TensorFlow. If imports fail: Runtime → Restart session, then Steps 1–5."],
            ["Model loading error", "Step 5: “model file could not be loaded (… deserialize …)”", "Keras in Colab is older/newer than the saving version (Keras 3.13.2). Run <i>!pip install -q tensorflow==2.21.0 keras==3.13.2</i>, Runtime → Restart session, then Steps 1–6. A truncated copy: delete the models folder and rerun Step 5."],
            ["Missing metadata", "Step 5: “metadata file … not found”", "Copy the matching <i>*_metadata.json</i> into the Drive model folder; the website refuses to guess thresholds or class mappings."],
            ["Port in use / app not starting", "Step 6: “did not start” or address already in use", "Run Step 9, then Step 6. Read the log printed by Step 8."],
            ["Proxy page blank", "Option A link loads forever or reconnects", "Use the same browser and account as Colab, allow third-party cookies for *.googleusercontent.com, or use Option B."],
            ["Tunnel error", "Option B: no URL / 502 / 1033", "Make sure Step 6 says running, rerun Step 7B. A new run gives a new URL. If GitHub download fails, retry later or use Option A."],
            ["Upload fails (403)", "Red error after choosing a photo", "Start Streamlit only with the Step 6 cell (it disables XSRF for proxied access); reload the page."],
            ["Session disconnected", "“Runtime disconnected”, link dead", "Reconnect and run Steps 1–7 (Section 9)."],
            ["Not enough RAM / GPU memory", "Session crashes, “Your session crashed after using all available RAM”, ResourceExhausted", "Runtime → Disconnect and delete runtime, close other Colab tabs, run again. All five models need ≈ 2–3 GB RAM. Try a CPU runtime if the GPU is full."],
            ["Very slow analysis", "first analysis takes minutes", "The first click loads the models (≈ 1–2 min); later analyses take seconds. Use a T4 GPU runtime."],
            ["Stage shows Unavailable", "result card grey “Unavailable”", "The model or its metadata is missing/unloadable — see Models &amp; About in the website and fix the file. This is by design: no fake results."]]
    story.append(table(rows, [32 * mm, 45 * mm, CW - 77 * mm]))

    # ---------------- 11 checklist
    story += [PageBreak(), H1("11. Final checklist")]
    checks = ["Colab and Drive use the same Google account.",
              "My Drive contains <b>dentalcaries</b> (own folder or shortcut).",
              "<b>Dental_Caries_AI_Website.zip</b> is in My Drive/dentalcaries (or My Drive).",
              "<b>pipeline_stage1_to_4_corrected</b> contains 5 .keras files + 5 *_metadata.json files.",
              "Runtime type is T4 GPU (or CPU, slower).",
              "Step 2 found the ZIP and the model folder.",
              "Step 3 printed OK for all five checked files.",
              "Step 4 printed the TensorFlow, Keras, Streamlit and ReportLab versions.",
              "Step 5 ended with “RESULT: all required models loaded”.",
              "Step 6 printed “Streamlit is running on port 8501”.",
              "Step 7 gave a link and the home page opens.",
              "The questionnaire gives the expected score (example: 9 / 13 HIGH).",
              "An analysis completes and the stage cards show their statuses.",
              "The PDF report downloads and opens.",
              "Step 9 was run (especially after using a public tunnel)."]
    story.append(table([["☐", c] for c in checks], [9 * mm, CW - 9 * mm], head=False))
    story += [Spacer(1, 6)]

    # ---------------- 12 appendix: verification status
    story += [H1("12. Appendix — what was tested")]
    story += bullets(["<b>Automated tests (44, all passing):</b> questionnaire scoring for all 648 answer combinations; upload validation "
                      "(empty, wrong type, fake content, truncated JPEG, too small, EXIF rotation); model loading from .keras files; "
                      "Stage 1 stop rules (HEALTHY, UNCERTAIN, continue-if-uncertain); FDI association (assigned, ambiguous, no tooth); "
                      "no-lesion and missing/corrupted model/metadata behaviour; class mappings taken from metadata; PDF/TXT/JSON/CSV/JPEG/ZIP "
                      "generation; Streamlit pages (headless AppTest).",
                      "<b>Notebook parity:</b> the notebook's own code (Sections 3, 6.1, 6.6, 6.7, 76, 82, executed from the reference copy) and "
                      "the website return identical results, annotations, tables and report text on the same models and images.",
                      "<b>Browser end-to-end:</b> the real Streamlit server was started and driven in Chromium: questionnaire → upload → "
                      "analyse → results → PDF download.",
                      "<b>Not verified here:</b> the website could not be run inside Google Colab itself, and the trained weights could not "
                      "be downloaded into the test environment (Google Drive downloads were blocked by its network policy). Tests therefore "
                      "used stand-in models with the same file format, shapes and metadata keys. Run Step 5 (model check) and the "
                      "Section 7 example in your own Colab session to confirm the real models. The README explains how to re-run the "
                      "parity test with your trained models."])
    story += [Spacer(1, 6), P(escape(config.ACADEMIC_DISCLAIMER), "small")]

    doc = Doc(out)
    doc.multiBuild(story)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "Dental_Caries_AI_Colab_Guide.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("wrote", build(out))
