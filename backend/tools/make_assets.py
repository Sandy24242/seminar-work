"""
Generates every decorative image in assets/ (original artwork drawn with Pillow — free to redistribute with the project).
Run:  python tools/make_assets.py
Decorative images only: no prediction or annotation shown on the website comes from these files.
"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
FONT_DIR = os.path.join(OUT, "fonts")
SS = 3  # supersampling factor

NAVY = (11, 37, 69); TEAL = (14, 124, 134); TEAL2 = (19, 163, 180); SKY = (205, 235, 240); LIGHT = (234, 246, 248)
WHITE = (255, 255, 255); ENAMEL = (250, 252, 255); DENTIN = (246, 228, 196); PULP = (232, 112, 120); GUM = (240, 160, 170)
BONE = (236, 226, 210); CARIES = (150, 104, 40); ORANGE = (255, 140, 0); GREEN = (40, 190, 90)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


def canvas(w, h, color=(0, 0, 0, 0)):
    return Image.new("RGBA", (w * SS, h * SS), color)


def finish(img, w, h):
    return img.resize((w, h), Image.LANCZOS)


def bezier(p0, p1, p2, p3, n=40):
    out = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        out.append((u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
                    u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def tooth_path(cx, cy, s):
    """Molar silhouette (crown with two cusps + two roots) centred at (cx, cy), height ~ 2*s."""
    P = lambda x, y: (cx + x * s, cy + y * s)
    pts = []
    pts += bezier(P(-0.80, -0.55), P(-0.85, -1.05), P(-0.35, -1.10), P(-0.20, -0.85))
    pts += bezier(P(-0.20, -0.85), P(-0.08, -0.72), P(0.08, -0.72), P(0.20, -0.85))
    pts += bezier(P(0.20, -0.85), P(0.35, -1.10), P(0.85, -1.05), P(0.80, -0.55))
    pts += bezier(P(0.80, -0.55), P(0.78, -0.10), P(0.62, 0.05), P(0.58, 0.30))
    pts += bezier(P(0.58, 0.30), P(0.52, 0.70), P(0.50, 1.00), P(0.32, 1.02))
    pts += bezier(P(0.32, 1.02), P(0.16, 1.03), P(0.14, 0.55), P(0.06, 0.32))
    pts += bezier(P(0.06, 0.32), P(0.02, 0.24), P(-0.02, 0.24), P(-0.06, 0.32))
    pts += bezier(P(-0.06, 0.32), P(-0.14, 0.55), P(-0.16, 1.03), P(-0.32, 1.02))
    pts += bezier(P(-0.32, 1.02), P(-0.50, 1.00), P(-0.52, 0.70), P(-0.58, 0.30))
    pts += bezier(P(-0.58, 0.30), P(-0.62, 0.05), P(-0.78, -0.10), P(-0.80, -0.55))
    return pts


def gradient(w, h, top, bottom, horizontal=False):
    g = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(g)
    n = w if horizontal else h
    for i in range(n):
        t = i / max(n - 1, 1)
        c = tuple(int(top[k] + (bottom[k] - top[k]) * t) for k in range(3)) + (255,)
        if horizontal:
            d.line([(i, 0), (i, h)], fill=c)
        else:
            d.line([(0, i), (w, i)], fill=c)
    return g


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    return m


def shadow(img, box, radius, offset=(0, 10), blur=28, alpha=60):
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x1, y1, x2, y2 = box
    ImageDraw.Draw(sh).rounded_rectangle([x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]], radius,
                                         fill=(11, 37, 69, alpha))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)))


# ------------------------------------------------------------------------------------------------------------- logo
def make_logo(size=512):
    S = size * SS
    img = canvas(size, size)
    bg = gradient(S, S, TEAL2, NAVY)
    img.paste(bg, (0, 0), rounded_mask(S, S, int(S * 0.22)))
    d = ImageDraw.Draw(img)
    d.polygon(tooth_path(S * 0.5, S * 0.52, S * 0.30), fill=WHITE)
    # AI "scan" ring + nodes
    r = S * 0.40
    d.arc([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], 200, 340, fill=(255, 255, 255, 170), width=int(S * 0.018))
    for a in (200, 270, 340):
        x = S / 2 + r * math.cos(math.radians(a)); y = S / 2 + r * math.sin(math.radians(a))
        d.ellipse([x - S * 0.028, y - S * 0.028, x + S * 0.028, y + S * 0.028], fill=SKY)
    # small caries dot being "found"
    d.ellipse([S * 0.56, S * 0.33, S * 0.64, S * 0.41], fill=CARIES)
    d.rounded_rectangle([S * 0.52, S * 0.29, S * 0.68, S * 0.45], int(S * 0.02), outline=(255, 210, 120), width=int(S * 0.012))
    return finish(img, size, size)


# ------------------------------------------------------------------------------------------------------------- hero
def make_hero(w=1600, h=900):
    img = canvas(w, h)
    W, H = w * SS, h * SS
    img.alpha_composite(gradient(W, H, (246, 251, 253), (214, 238, 244)))
    d = ImageDraw.Draw(img)
    # dotted grid
    for x in range(0, W, 46 * SS):
        for y in range(0, H, 46 * SS):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(14, 124, 134, 40))
    # soft circles
    for (cx, cy, r, a) in ((0.72, 0.45, 0.42, 70), (0.92, 0.85, 0.22, 60), (0.1, 0.9, 0.18, 40)):
        d.ellipse([W * cx - H * r, H * cy - H * r, W * cx + H * r, H * cy + H * r], fill=(19, 163, 180, a))
    # monitor card
    bx = [W * 0.50, H * 0.14, W * 0.94, H * 0.80]
    shadow(img, [int(v) for v in bx], 40 * SS)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(bx, 36 * SS, fill=WHITE)
    d.rounded_rectangle([bx[0], bx[1], bx[2], bx[1] + 56 * SS], 36 * SS, fill=NAVY)
    d.rectangle([bx[0], bx[1] + 30 * SS, bx[2], bx[1] + 56 * SS], fill=NAVY)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([bx[0] + (28 + i * 30) * SS, bx[1] + 20 * SS, bx[0] + (44 + i * 30) * SS, bx[1] + 36 * SS], fill=c)
    # illustrated "intraoral view" inside the card (decorative drawing)
    ix = [bx[0] + 30 * SS, bx[1] + 80 * SS, bx[2] - 30 * SS, bx[3] - 120 * SS]
    d.rounded_rectangle(ix, 22 * SS, fill=(196, 92, 104))
    d.rounded_rectangle([ix[0], ix[1], ix[2], ix[1] + (ix[3] - ix[1]) * 0.35], 22 * SS, fill=(222, 128, 140))
    tw = (ix[2] - ix[0]) / 4.3
    for i in range(4):
        cx = ix[0] + tw * (0.75 + i * 0.95); cy = ix[1] + (ix[3] - ix[1]) * 0.60
        d.rounded_rectangle([cx - tw * 0.42, cy - tw * 0.62, cx + tw * 0.42, cy + tw * 0.55], int(tw * 0.25), fill=ENAMEL)
    lx = ix[0] + tw * (0.75 + 2 * 0.95); ly = ix[1] + (ix[3] - ix[1]) * 0.58
    d.ellipse([lx - tw * 0.12, ly - tw * 0.10, lx + tw * 0.14, ly + tw * 0.14], fill=CARIES)
    d.rectangle([lx - tw * 0.24, ly - tw * 0.22, lx + tw * 0.26, ly + tw * 0.26], outline=GREEN, width=5 * SS)
    for i in range(4):
        cx = ix[0] + tw * (0.75 + i * 0.95); cy = ix[1] + (ix[3] - ix[1]) * 0.60
        d.rectangle([cx - tw * 0.46, cy - tw * 0.66, cx + tw * 0.46, cy + tw * 0.59], outline=ORANGE, width=3 * SS)
    # scan line
    sy = ix[1] + (ix[3] - ix[1]) * 0.30
    d.line([ix[0], sy, ix[2], sy], fill=(19, 163, 180, 200), width=4 * SS)
    # stats under the image
    yb = ix[3] + 26 * SS
    for i, (lab, wv) in enumerate((("Stage 1", 0.92), ("Stage 2", 0.78), ("Stage 3", 0.64), ("Stage 4", 0.55))):
        x0 = ix[0] + i * (ix[2] - ix[0]) / 4
        d.text((x0, yb), lab, font=font(15 * SS, True), fill=NAVY)
        d.rounded_rectangle([x0, yb + 26 * SS, x0 + (ix[2] - ix[0]) / 4 - 24 * SS, yb + 36 * SS], 5 * SS, fill=SKY)
        d.rounded_rectangle([x0, yb + 26 * SS, x0 + ((ix[2] - ix[0]) / 4 - 24 * SS) * wv, yb + 36 * SS], 5 * SS, fill=TEAL)
    # big tooth with magnifier on the left-bottom of the card
    tx, ty, ts = W * 0.26, H * 0.60, H * 0.22
    shadow(img, [int(tx - ts), int(ty - ts), int(tx + ts), int(ty + ts)], int(ts), blur=40, alpha=50)
    d = ImageDraw.Draw(img)
    d.ellipse([tx - ts * 1.18, ty - ts * 1.18, tx + ts * 1.18, ty + ts * 1.18], fill=WHITE)
    d.polygon(tooth_path(tx, ty + ts * 0.05, ts * 0.78), fill=(214, 238, 244))
    d.polygon(tooth_path(tx, ty, ts * 0.78), fill=TEAL)
    mx, my, mr = tx + ts * 0.62, ty - ts * 0.55, ts * 0.42
    d.ellipse([mx - mr, my - mr, mx + mr, my + mr], outline=NAVY, width=12 * SS, fill=(255, 255, 255, 140))
    d.line([mx + mr * 0.7, my + mr * 0.7, mx + mr * 1.6, my + mr * 1.6], fill=NAVY, width=18 * SS)
    # sparkles
    for (px, py, pr) in ((0.30, 0.25, 10), (0.36, 0.40, 6), (0.25, 0.5, 8)):
        x, y, r = W * px, H * py, pr * SS
        d.polygon([(x, y - 3 * r), (x + r, y), (x, y + 3 * r), (x - r, y)], fill=TEAL2)
        d.polygon([(x - 3 * r, y), (x, y + r), (x + 3 * r, y), (x, y - r)], fill=TEAL2)
    return finish(img, w, h).convert("RGB")


# ------------------------------------------------------------------------------------------------------------- anatomy
def make_anatomy(w=900, h=1000):
    img = canvas(w, h, WHITE + (255,))
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(img)
    cx, cy, s = W * 0.34, H * 0.55, H * 0.30
    # bone + gum
    d.rounded_rectangle([cx - s * 1.0, cy + s * 0.05, cx + s * 1.0, H * 0.98], 30 * SS, fill=BONE)
    d.polygon([(cx - s * 1.0, cy + s * 0.15), (cx - s * 0.75, cy - s * 0.05), (cx - s * 0.55, cy + s * 0.05),
               (cx + s * 0.55, cy + s * 0.05), (cx + s * 0.75, cy - s * 0.05), (cx + s * 1.0, cy + s * 0.15),
               (cx + s * 1.0, cy + s * 0.45), (cx - s * 1.0, cy + s * 0.45)], fill=GUM)
    # enamel / dentin / pulp layers (scaled tooth outlines)
    d.polygon(tooth_path(cx, cy, s), fill=(232, 238, 244))
    d.polygon(tooth_path(cx, cy + s * 0.03, s * 0.93), fill=ENAMEL)
    d.polygon(tooth_path(cx, cy + s * 0.10, s * 0.80), fill=DENTIN)
    pulp = tooth_path(cx, cy + s * 0.25, s * 0.45)
    d.polygon(pulp, fill=PULP)
    # caries lesion through enamel into dentin
    d.ellipse([cx - s * 0.48, cy - s * 0.92, cx - s * 0.26, cy - s * 0.70], fill=CARIES)
    d.ellipse([cx - s * 0.44, cy - s * 0.80, cx - s * 0.30, cy - s * 0.62], fill=(120, 80, 30))
    # labels
    labels = [("Caries lesion", (cx - s * 0.37, cy - s * 0.80)), ("Enamel", (cx + s * 0.80, cy - s * 0.62)),
              ("Dentin", (cx + s * 0.55, cy - s * 0.30)), ("Pulp", (cx + s * 0.15, cy + s * 0.10)),
              ("Gum (gingiva)", (cx + s * 0.88, cy + s * 0.25)), ("Root", (cx + s * 0.42, cy + s * 0.78)),
              ("Bone", (cx + s * 0.85, cy + s * 1.05))]
    tx = W * 0.70
    for i, (lab, (px, py)) in enumerate(labels):
        ty = H * (0.17 + i * 0.115)
        d.line([px, py, tx - 14 * SS, ty + 16 * SS], fill=NAVY, width=3 * SS)
        d.ellipse([px - 8 * SS, py - 8 * SS, px + 8 * SS, py + 8 * SS], fill=TEAL)
        d.text((tx, ty), lab, font=font(28 * SS, True), fill=NAVY)
    d.text((W * 0.05, H * 0.03), "Tooth anatomy", font=font(40 * SS, True), fill=NAVY)
    d.text((W * 0.05, H * 0.03 + 52 * SS), "illustration (not to scale)", font=font(22 * SS), fill=(90, 110, 130))
    return finish(img, w, h).convert("RGB")


# ------------------------------------------------------------------------------------------------------------- icons
def icon_base(size=256, colors=(TEAL2, TEAL)):
    S = size * SS
    img = canvas(size, size)
    img.paste(gradient(S, S, *colors), (0, 0), rounded_mask(S, S, int(S * 0.26)))
    return img, ImageDraw.Draw(img), S


def make_icons(size=256):
    icons = {}
    # Stage 1 — detection (tooth + check)
    img, d, S = icon_base(size)
    d.polygon(tooth_path(S * 0.45, S * 0.52, S * 0.28), fill=WHITE)
    d.ellipse([S * 0.56, S * 0.56, S * 0.86, S * 0.86], fill=NAVY)
    d.line([(S * 0.63, S * 0.71), (S * 0.69, S * 0.78), (S * 0.80, S * 0.64)], fill=WHITE, width=int(S * 0.04), joint="curve")
    icons["icon_stage1"] = img
    # Stage 2 — FDI tooth identification (tooth + number badge)
    img, d, S = icon_base(size)
    d.polygon(tooth_path(S * 0.42, S * 0.54, S * 0.27), fill=WHITE)
    d.rounded_rectangle([S * 0.56, S * 0.18, S * 0.88, S * 0.42], int(S * 0.06), fill=ORANGE)
    d.text((S * 0.72, S * 0.30), "16", font=font(int(S * 0.15), True), fill=NAVY, anchor="mm")
    icons["icon_stage2"] = img
    # Localization — crosshair box
    img, d, S = icon_base(size)
    d.polygon(tooth_path(S * 0.5, S * 0.52, S * 0.30), fill=WHITE)
    d.ellipse([S * 0.40, S * 0.30, S * 0.52, S * 0.42], fill=CARIES)
    d.rectangle([S * 0.32, S * 0.22, S * 0.60, S * 0.50], outline=GREEN, width=int(S * 0.03))
    for (x1, y1, x2, y2) in ((S * 0.46, S * 0.12, S * 0.46, S * 0.22), (S * 0.46, S * 0.50, S * 0.46, S * 0.60),
                             (S * 0.22, S * 0.36, S * 0.32, S * 0.36), (S * 0.60, S * 0.36, S * 0.70, S * 0.36)):
        d.line([x1, y1, x2, y2], fill=NAVY, width=int(S * 0.025))
    icons["icon_localization"] = img
    # Stage 3 — G.V. Black (roman numerals)
    img, d, S = icon_base(size, (TEAL, NAVY))
    for i, t in enumerate(("I", "II", "III", "IV", "V", "VI")):
        x = S * (0.27 + (i % 3) * 0.23); y = S * (0.36 + (i // 3) * 0.28)
        d.rounded_rectangle([x - S * 0.10, y - S * 0.11, x + S * 0.10, y + S * 0.11], int(S * 0.04),
                            fill=WHITE if i != 3 else ORANGE)
        d.text((x, y), t, font=font(int(S * 0.10), True), fill=NAVY, anchor="mm")
    icons["icon_stage3"] = img
    # Stage 4 — ICDAS scale bars
    img, d, S = icon_base(size, (TEAL, NAVY))
    for i in range(6):
        x = S * (0.18 + i * 0.115); hgt = S * (0.12 + i * 0.075)
        d.rounded_rectangle([x, S * 0.78 - hgt, x + S * 0.08, S * 0.78], int(S * 0.02), fill=WHITE if i != 4 else ORANGE)
        d.text((x + S * 0.04, S * 0.86), str(i + 1), font=font(int(S * 0.07), True), fill=WHITE, anchor="mm")
    icons["icon_stage4"] = img
    # Questionnaire — clipboard
    img, d, S = icon_base(size, (TEAL2, TEAL))
    d.rounded_rectangle([S * 0.26, S * 0.20, S * 0.74, S * 0.84], int(S * 0.05), fill=WHITE)
    d.rounded_rectangle([S * 0.38, S * 0.14, S * 0.62, S * 0.26], int(S * 0.04), fill=NAVY)
    for i in range(4):
        y = S * (0.36 + i * 0.12)
        d.ellipse([S * 0.32, y - S * 0.03, S * 0.38, y + S * 0.03], fill=TEAL if i < 3 else ORANGE)
        d.rounded_rectangle([S * 0.43, y - S * 0.015, S * 0.66, y + S * 0.015], int(S * 0.01), fill=(150, 170, 185))
    icons["icon_questionnaire"] = img
    # Report — document with download arrow
    img, d, S = icon_base(size, (TEAL2, TEAL))
    d.polygon([(S * 0.28, S * 0.16), (S * 0.60, S * 0.16), (S * 0.74, S * 0.30), (S * 0.74, S * 0.84), (S * 0.28, S * 0.84)], fill=WHITE)
    d.polygon([(S * 0.60, S * 0.16), (S * 0.74, S * 0.30), (S * 0.60, S * 0.30)], fill=SKY)
    for i in range(3):
        y = S * (0.38 + i * 0.09)
        d.rounded_rectangle([S * 0.36, y, S * 0.66, y + S * 0.03], int(S * 0.01), fill=(150, 170, 185))
    d.line([S * 0.51, S * 0.62, S * 0.51, S * 0.76], fill=TEAL, width=int(S * 0.04))
    d.polygon([(S * 0.43, S * 0.70), (S * 0.59, S * 0.70), (S * 0.51, S * 0.79)], fill=TEAL)
    icons["icon_report"] = img
    return {k: finish(v, size, size) for k, v in icons.items()}


# ------------------------------------------------------------------------------------------------------------- workflow
def make_workflow(w=2200, h=520):
    img = canvas(w, h, WHITE + (255,))
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(img)
    steps = [("Questionnaire", "habit risk score", "icon_questionnaire"),
             ("Stage 1", "healthy vs caries", "icon_stage1"),
             ("Stage 2", "FDI tooth numbers", "icon_stage2"),
             ("Localization", "lesion boxes", "icon_localization"),
             ("Stage 3", "G.V. Black class", "icon_stage3"),
             ("Stage 4", "ICDAS score", "icon_stage4"),
             ("Report", "PDF · TXT · JSON · CSV", "icon_report")]
    icons = make_icons(256)
    n = len(steps); gap = W / n
    for i, (t, sub, ic) in enumerate(steps):
        cx = gap * (i + 0.5); cy = H * 0.40
        r = H * 0.18
        shadow(img, [int(cx - r), int(cy - r), int(cx + r), int(cy + r)], int(r * 0.4), blur=20, alpha=45)
        d = ImageDraw.Draw(img)
        icon = icons[ic].resize((int(r * 2), int(r * 2)), Image.LANCZOS)
        img.alpha_composite(icon, (int(cx - r), int(cy - r)))
        d.text((cx, H * 0.75), t, font=font(int(H * 0.06), True), fill=NAVY, anchor="mm")
        d.text((cx, H * 0.86), sub, font=font(int(H * 0.042)), fill=(80, 100, 120), anchor="mm")
        if i < n - 1:
            x1, x2 = cx + r + 14 * SS, gap * (i + 1.5) - r - 14 * SS
            d.line([x1, cy, x2 - 18 * SS, cy], fill=TEAL2, width=7 * SS)
            d.polygon([(x2, cy), (x2 - 26 * SS, cy - 16 * SS), (x2 - 26 * SS, cy + 16 * SS)], fill=TEAL2)
    return finish(img, w, h).convert("RGB")


def make_clinical_example(src):
    """Crop of the upload preview recorded in the user's own notebook run (Section 82.1 output, image UFV72.jpg)."""
    im = Image.open(src).convert("RGB")
    # matplotlib figure: title on top, white margins around the axes
    import numpy as np
    a = np.asarray(im).astype(int)
    nonwhite = (np.abs(a - 255).sum(2) > 30)
    rows = np.where(nonwhite.mean(1) > 0.5)[0]; cols = np.where(nonwhite.mean(0) > 0.5)[0]
    return im.crop((cols.min() + 2, rows.min() + 2, cols.max() - 1, rows.max() - 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    logo = make_logo(512)
    logo.save(os.path.join(OUT, "logo.png"))
    logo.resize((64, 64), Image.LANCZOS).save(os.path.join(OUT, "favicon.png"))
    make_hero().save(os.path.join(OUT, "hero_dental_ai.jpg"), quality=90)
    make_anatomy().save(os.path.join(OUT, "tooth_anatomy.png"))
    os.makedirs(os.path.join(OUT, "icons"), exist_ok=True)
    for k, v in make_icons(256).items():
        v.save(os.path.join(OUT, "icons", f"{k}.png"))
    make_workflow().save(os.path.join(OUT, "workflow_stages.png"))
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        make_clinical_example(sys.argv[1]).save(os.path.join(OUT, "clinical_caries_example.jpg"), quality=92)
    print("assets written to", OUT)


if __name__ == "__main__":
    main()
