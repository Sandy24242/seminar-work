"""
Annotated result image — notebook Section 76 `draw_result` / `_put_label`, same colours and label format:
teeth = orange boxes with the FDI number ('?' when uncertain); lesions = green boxes 'L<id> T<tooth>: <G.V. Black> | ICDAS <score>'.
Only real detections from the uploaded photograph are drawn.
"""
import cv2

TOOTH_COLOR, LESION_COLOR = (255, 140, 0), (0, 255, 0)


def _put_label(img, text, x, y, color, fs, ft, above=True):
    font = cv2.FONT_HERSHEY_SIMPLEX; H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
    x = int(max(0, min(x, W - tw - 6)))                     # keep the label inside the image
    ty = y - 6 if above and y - th - 10 > 0 else min(y + th + 6, H - 2)
    cv2.rectangle(img, (x, ty - th - 4), (min(x + tw + 6, W), ty + 4), color, -1)
    cv2.putText(img, text, (x + 3, ty), font, fs, (0, 0, 0), ft, cv2.LINE_AA)


def _lesion_text(les):
    tooth = les.get("fdi_tooth") or "?"
    g = les.get("gvb_display") or "n/a"
    i = les.get("icdas")
    return f"L{les['lesion']} T{tooth}: {g} | ICDAS {i if i is not None else 'n/a'}"


def draw_result(rgb, res, color=LESION_COLOR, show_teeth=True):
    """Tooth boxes (orange, FDI number or '?') and lesion boxes (green, lesion id -> tooth | G.V. Black | ICDAS)."""
    out = rgb.copy(); H, W = out.shape[:2]; s = max(H, W) / 1000.0
    th, fs, ft = max(2, round(3 * s)), max(0.5, 0.8 * s), max(1, round(2 * s))
    if show_teeth:
        for t in res.get("teeth", []):
            x1, y1, x2, y2 = t["box_xyxy_px"]; cv2.rectangle(out, (x1, y1), (x2, y2), TOOTH_COLOR, max(1, th // 2))
            _put_label(out, t["fdi"] if not t["fdi_uncertain"] else "?", x1, y1 + 2 * th + int(22 * s), TOOTH_COLOR,
                       0.8 * fs, max(1, ft - 1), above=True)
    for les in res.get("lesions", []):
        x1, y1, x2, y2 = les["box_xyxy_px"]; cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
        _put_label(out, _lesion_text(les), x1, y1 if y1 > 0 else y2, color, fs, ft, above=y1 > 0)
    return out
