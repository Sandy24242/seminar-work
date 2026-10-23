"""
Dataset validation utilities: image integrity, YOLO label parsing, view codes, duplicates, ICDAS folder counts.

Important distinction: folder names such as 'Upper Occlusal View' (or filename codes UOV/LFV/...) are PHOTOGRAPHIC VIEWS,
not disease classes. Disease labels come only from annotation class IDs (YOLO .txt) or from explicitly labelled class
folders (ICDAS_<k>, healthy/caries roots). Nothing in this module infers a label from a view.
"""
import os
import re
import zipfile
from collections import Counter, defaultdict

from PIL import Image, UnidentifiedImageError

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

VIEW_FOLDERS = {
    "upper front view": "UFV", "upper left view": "ULV", "upper right view": "URV", "upper occlusal view": "UOV",
    "lower front view": "LFV", "lower left view": "LLV", "lower right view": "LRV", "lower occlusal view": "LOV",
}
VIEW_CODES = set(VIEW_FOLDERS.values())
_VIEW_RE = re.compile(r"(?<![A-Z])(UFV|ULV|URV|UOV|LFV|LLV|LRV|LOV)(?![A-Z])")


def view_of(path):
    """Photographic view code from a folder name or filename, or None. Never a disease label."""
    for part in reversed(path.replace("\\", "/").split("/")):
        code = VIEW_FOLDERS.get(part.strip().lower())
        if code:
            return code
        m = _VIEW_RE.search(part.upper())
        if m:
            return m.group(1)
    return None


def list_images(root):
    out = []
    for dp, _dn, fn in os.walk(root):
        out += [os.path.join(dp, f) for f in fn if f.lower().endswith(IMG_EXT)]
    return sorted(out)


def check_image(path):
    """(ok, info_or_error). Full decode, not only header verification."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            return True, {"format": im.format, "size": im.size, "mode": im.mode}
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        return False, f"{type(e).__name__}: {e}"


def parse_yolo_file(path):
    """Returns (boxes, problems). Each box = (class_id, xc, yc, w, h), normalised 0-1."""
    boxes, problems = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                problems.append(f"line {i}: expected 5 values, got {len(parts)}")
                continue
            try:
                c = int(float(parts[0])); xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                problems.append(f"line {i}: non-numeric value")
                continue
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                problems.append(f"line {i}: coordinates outside 0-1")
            boxes.append((c, xc, yc, w, h))
    return boxes, problems


def read_class_names(dataset_root):
    """classes.txt (one name per line) or data.yaml 'names:'; None if neither exists (mapping then NOT verifiable)."""
    p = os.path.join(dataset_root, "classes.txt")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    p = os.path.join(dataset_root, "data.yaml")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            m = re.search(r"names:\s*\[(.*?)\]", f.read(), re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    return None


def inventory_yolo(dataset_root, images_sub="images", labels_sub="labels", exclude_dirs=()):
    """Pairs images/<...>/x.jpg with labels/<...>/x.txt (or same-folder .txt) and summarises everything."""
    img_root = os.path.join(dataset_root, images_sub)
    if not os.path.isdir(img_root):
        img_root = dataset_root
    images = [p for p in list_images(img_root) if not any(f"/{d}/" in p.replace("\\", "/") for d in exclude_dirs)]
    names = read_class_names(dataset_root)
    per_class, per_view, problems, corrupt = Counter(), Counter(), [], []
    n_labelled = n_unlabelled = n_empty = 0
    stem_map = {}
    for p in images:
        rel = os.path.relpath(p, img_root)
        cands = [os.path.join(dataset_root, labels_sub, os.path.splitext(rel)[0] + ".txt"), os.path.splitext(p)[0] + ".txt"]
        lab = next((c for c in cands if os.path.isfile(c)), None)
        ok, info = check_image(p)
        if not ok:
            corrupt.append({"image": rel, "error": info})
        per_view[view_of(rel) or "unknown"] += 1
        stem_map[os.path.splitext(rel)[0]] = p
        if lab is None:
            n_unlabelled += 1
            continue
        boxes, probs = parse_yolo_file(lab)
        problems += [f"{os.path.relpath(lab, dataset_root)}: {x}" for x in probs]
        if not boxes:
            n_empty += 1
        n_labelled += 1
        for b in boxes:
            per_class[names[b[0]] if names and 0 <= b[0] < len(names) else f"id_{b[0]}"] += 1
    lab_root = os.path.join(dataset_root, labels_sub)
    orphans = []
    if os.path.isdir(lab_root):
        for dp, _dn, fn in os.walk(lab_root):
            for f in fn:
                if f.endswith(".txt"):
                    stem = os.path.splitext(os.path.relpath(os.path.join(dp, f), lab_root))[0]
                    if stem not in stem_map:
                        orphans.append(stem)
    return {"n_images": len(images), "n_labelled": n_labelled, "n_unlabelled": n_unlabelled, "n_empty_labels": n_empty,
            "class_names_source": "classes.txt/data.yaml" if names else "MISSING (class-ID mapping not verifiable)",
            "class_names": names, "boxes_per_class": dict(per_class), "images_per_view": dict(per_view),
            "corrupt_images": corrupt, "label_problems": problems[:200], "n_label_problems": len(problems),
            "orphan_labels": orphans[:200], "n_orphan_labels": len(orphans)}


def inventory_class_folders(root, pattern=r"^(?:ICDAS[_ ]?)?([0-6])$"):
    """Counts images per class folder whose NAME matches pattern (e.g. ICDAS_3 -> '3'). Other folders are reported apart."""
    rx = re.compile(pattern, re.I)
    counts, other, corrupt = Counter(), Counter(), []
    for p in list_images(root):
        rel = os.path.relpath(p, root)
        label = next((rx.match(part).group(1) for part in rel.replace("\\", "/").split("/")[:-1] if rx.match(part)), None)
        ok, info = check_image(p)
        if not ok:
            corrupt.append({"image": rel, "error": info})
        if label is None:
            other[os.path.dirname(rel) or "."] += 1
        else:
            counts[label] += 1
    return {"images_per_class": dict(sorted(counts.items())), "images_outside_class_folders": dict(other),
            "corrupt_images": corrupt}


def dhash(path, size=8):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def find_duplicates(paths, max_hamming=0):
    """Groups of (near-)identical images by 64-bit dHash. max_hamming=0 -> exact visual duplicates only."""
    hashes = {}
    for p in paths:
        try:
            hashes[p] = dhash(p)
        except (OSError, ValueError):
            continue
    if max_hamming == 0:
        groups = defaultdict(list)
        for p, h in hashes.items():
            groups[h].append(p)
        return [g for g in groups.values() if len(g) > 1]
    items, seen, out = list(hashes.items()), set(), []
    for i, (p, h) in enumerate(items):
        if p in seen:
            continue
        g = [p] + [q for q, k in items[i + 1:] if q not in seen and bin(h ^ k).count("1") <= max_hamming]
        if len(g) > 1:
            seen.update(g); out.append(g)
    return out


def zip_integrity(path):
    """(ok, n_members, first_bad_member). Uses zipfile.testzip (CRC of every member)."""
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        return bad is None, len(z.infolist()), bad
