"""
Read-only audit of the project's dataset archives (ZIP files or extracted folders). Nothing is extracted or modified.

    python scripts/audit_datasets.py healthy.zip caries.zip "lowoccyolo (4).zip" icdascl.zip caries_FDI_dataset_final.zip
    python scripts/audit_datasets.py /content/drive/MyDrive/dentalcaries/*.zip --near-dup --out audit.json --md audit.md

Per dataset it reports what is actually in the files:
  * file types, folders and image counts per folder (folder names are the only class labels some datasets have)
  * images that cannot be decoded (corrupt / truncated / not really an image), image formats, sizes, colour modes
  * exact duplicates (MD5) inside a dataset and ACROSS datasets (e.g. the same photo in "healthy" and "caries")
  * optional near-duplicates (64-bit dHash, Hamming <= --hamming), the same measure the notebook's split uses
  * YOLO labels (*.txt "cls cx cy w h"): class-id counts, malformed lines, out-of-range boxes, images without labels,
    labels without images, class names from classes.txt / data.yaml if present
  * COCO-style JSON annotations: categories and annotation counts
  * ICDAS class folders (named "ICDAS_<k>" or "<k>", k = 0..6), the same rule as the notebook's Section 7.2
  * whether any patient identifier exists (file / folder names or CSV columns containing "patient")

It does not decide which label a file "should" have and never fills in missing labels.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFile

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
ICDAS_FOLDER_RE = re.compile(r"^(?:icdas)?[\s_\-]*([0-6])$", re.I)      # notebook Section 7.2
PATIENT_RE = re.compile(r"patient|subject", re.I)
Image.MAX_IMAGE_PIXELS = 200_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False


# ---------------------------------------------------------------- sources (zip or folder, read-only)
class Source:
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(os.path.normpath(path))
        self.is_zip = os.path.isfile(path) and zipfile.is_zipfile(path)
        if self.is_zip:
            self.zf = zipfile.ZipFile(path)
            self.files = [i.filename for i in self.zf.infolist() if not i.is_dir() and not _junk(i.filename)]
        elif os.path.isdir(path):
            self.files = []
            for dp, dn, fn in os.walk(path):
                dn[:] = sorted(d for d in dn if not d.startswith("."))
                self.files += [os.path.relpath(os.path.join(dp, f), path).replace(os.sep, "/") for f in sorted(fn) if not _junk(f)]
        else:
            raise FileNotFoundError(f"{path} is neither a ZIP file nor a folder")

    def read(self, rel):
        if self.is_zip:
            return self.zf.read(rel)
        with open(os.path.join(self.path, rel), "rb") as f:
            return f.read()


def _junk(name):
    parts = name.replace("\\", "/").split("/")
    return any(p.startswith("__MACOSX") or p in (".DS_Store", "Thumbs.db") or (p.startswith("._")) for p in parts)


def dhash64(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


# ---------------------------------------------------------------- per-dataset audit
def audit_source(src, check_images=True, want_hash=False):
    files = src.files
    ext = Counter(os.path.splitext(f)[1].lower() or "<none>" for f in files)
    images = [f for f in files if f.lower().endswith(IMG_EXT)]
    per_folder = Counter(os.path.dirname(f) or "." for f in images)
    rep = {"dataset": src.name, "kind": "zip" if src.is_zip else "folder", "n_files": len(files), "n_images": len(images),
           "extensions": dict(ext.most_common()), "top_level": sorted({f.split("/")[0] for f in files})[:50],
           "images_per_folder": dict(sorted(per_folder.items()))}

    # ---- images: decode, formats, MD5, dHash
    md5s, hashes, bad, fmt, mode, sizes = {}, {}, [], Counter(), Counter(), []
    if check_images:
        for i, f in enumerate(images, 1):
            try:
                data = src.read(f)
                md5s[f] = hashlib.md5(data).hexdigest()
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    fmt[im.format] += 1; mode[im.mode] += 1; sizes.append(im.size)
                    if want_hash:
                        hashes[f] = dhash64(im)
            except Exception as e:
                bad.append({"file": f, "error": f"{type(e).__name__}: {e}"[:200]})
            if i % 2000 == 0:
                print(f"    {src.name}: {i:,}/{len(images):,} images checked", flush=True)
        groups = defaultdict(list)
        for f, h in md5s.items():
            groups[h].append(f)
        dup = [g for g in groups.values() if len(g) > 1]
        wh = np.array(sizes) if sizes else np.zeros((0, 2))
        rep.update(
            n_unreadable=len(bad), unreadable=bad[:200], formats=dict(fmt), colour_modes=dict(mode),
            size_px={"min_w": int(wh[:, 0].min()), "max_w": int(wh[:, 0].max()), "min_h": int(wh[:, 1].min()),
                     "max_h": int(wh[:, 1].max()), "median_w": float(np.median(wh[:, 0])),
                     "median_h": float(np.median(wh[:, 1]))} if len(wh) else None,
            exact_duplicate_groups=len(dup), exact_duplicate_extra_copies=sum(len(g) - 1 for g in dup),
            exact_duplicate_examples=dup[:20],
            exact_duplicates_across_folders=sum(1 for g in dup if len({os.path.dirname(x) for x in g}) > 1))

    # ---- ICDAS folders
    icdas = Counter()
    for folder, n in per_folder.items():
        m = ICDAS_FOLDER_RE.match(os.path.basename(folder))
        if m:
            icdas[int(m.group(1))] += n
    if icdas:
        rep["icdas_class_folders"] = {f"ICDAS_{k}": icdas[k] for k in sorted(icdas)}

    # ---- YOLO labels
    txts = [f for f in files if f.lower().endswith(".txt") and os.path.basename(f).lower() not in ("classes.txt", "readme.txt")]
    yolo = audit_yolo(src, txts, images, files) if txts else None
    if yolo and yolo["n_label_files_parsed_as_yolo"]:
        rep["yolo"] = yolo
    other_txt = [f for f in txts if not (yolo and f in yolo["_yolo_files"])]
    if yolo:
        yolo.pop("_yolo_files", None)
    if other_txt:
        rep["other_text_files"] = other_txt[:30]

    # ---- COCO / other JSON, CSV
    for f in files:
        fl = f.lower()
        if fl.endswith(".json"):
            rep.setdefault("json_files", []).append(summarise_json(src, f))
        elif fl.endswith(".csv"):
            rep.setdefault("csv_files", []).append(summarise_csv(src, f))

    # ---- patient identifiers
    hits = [f for f in files if PATIENT_RE.search(f)]
    csv_cols = [c for j in rep.get("csv_files", []) for c in j.get("columns", []) if PATIENT_RE.search(c)]
    json_keys = [k for j in rep.get("json_files", []) for k in j.get("patient_keys", [])]
    rep["patient_ids"] = {"found": bool(hits or csv_cols or json_keys), "paths_mentioning_patient": hits[:20],
                          "csv_columns": csv_cols, "json_keys": json_keys}
    return rep, md5s, hashes


def audit_yolo(src, txts, images, files):
    stems = {os.path.splitext(f)[0].split("/")[-1].lower() for f in images}
    cls, bad_lines, out_of_range, empty, parsed = Counter(), [], 0, 0, []
    boxes_per_image = []
    for f in txts:
        try:
            lines = src.read(f).decode("utf-8", "replace").splitlines()
        except Exception:
            continue
        rows = [ln.split() for ln in lines if ln.strip()]
        if rows and not all(len(r) >= 5 and _num(r[:5]) for r in rows):
            if not any(len(r) >= 5 and _num(r[:5]) for r in rows):
                continue                                        # not a YOLO label file at all
        parsed.append(f)
        if not rows:
            empty += 1
        n = 0
        for j, r in enumerate(rows):
            if len(r) < 5 or not _num(r[:5]):
                bad_lines.append(f"{f}:{j + 1}: {' '.join(r)[:80]}")
                continue
            c, *xywh = r[:5]
            cls[int(float(c))] += 1; n += 1
            x, y, w, h = map(float, xywh)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                out_of_range += 1
        boxes_per_image.append(n)
    lbl_stems = {os.path.splitext(f)[0].split("/")[-1].lower() for f in parsed}
    names = None
    for f in files:
        b = os.path.basename(f).lower()
        if b == "classes.txt":
            names = [x.strip() for x in src.read(f).decode("utf-8", "replace").splitlines() if x.strip()]
        elif b in ("data.yaml", "dataset.yaml") and names is None:
            txt = src.read(f).decode("utf-8", "replace")
            m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
            if m:
                names = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
            else:
                names = re.findall(r"^\s*\d+\s*:\s*(.+?)\s*$", txt, re.M) or None
    return {"n_label_files_parsed_as_yolo": len(parsed), "class_id_counts": dict(sorted(cls.items())),
            "class_names_file": names, "n_boxes": int(sum(cls.values())), "malformed_lines": len(bad_lines),
            "malformed_examples": bad_lines[:20], "boxes_out_of_range": out_of_range, "empty_label_files": empty,
            "max_boxes_per_image": max(boxes_per_image, default=0),
            "images_without_label_file": len(stems - lbl_stems), "label_files_without_image": len(lbl_stems - stems),
            "_yolo_files": set(parsed)}


def _num(xs):
    try:
        [float(x) for x in xs]
        return True
    except ValueError:
        return False


def summarise_json(src, f):
    out = {"file": f}
    try:
        d = json.loads(src.read(f).decode("utf-8", "replace"))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:150]
        return out
    if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
        cats = {c.get("id"): c.get("name") for c in d["categories"]}
        cnt = Counter(cats.get(a.get("category_id"), a.get("category_id")) for a in d["annotations"])
        out.update(format="COCO", n_images=len(d["images"]), n_annotations=len(d["annotations"]),
                   categories=len(cats), annotations_per_category=dict(cnt.most_common()))
    else:
        out["top_level_keys"] = list(d)[:30] if isinstance(d, dict) else f"list[{len(d)}]"
    keys = set()
    _collect_keys(d, keys, 0)
    out["patient_keys"] = sorted(k for k in keys if PATIENT_RE.search(k))
    return out


def _collect_keys(o, keys, depth):
    if depth > 4:
        return
    if isinstance(o, dict):
        for k, v in list(o.items())[:200]:
            keys.add(str(k)); _collect_keys(v, keys, depth + 1)
    elif isinstance(o, list):
        for v in o[:20]:
            _collect_keys(v, keys, depth + 1)


def summarise_csv(src, f):
    try:
        rows = list(csv.reader(io.StringIO(src.read(f).decode("utf-8", "replace"))))
    except Exception as e:
        return {"file": f, "error": f"{type(e).__name__}: {e}"[:150]}
    return {"file": f, "columns": rows[0] if rows else [], "n_rows": max(len(rows) - 1, 0)}


# ---------------------------------------------------------------- cross-dataset checks
def cross_exact(all_md5):
    where = defaultdict(set)
    for ds, m in all_md5.items():
        for h in set(m.values()):
            where[h].add(ds)
    pairs = Counter(tuple(sorted(s)) for s in where.values() if len(s) > 1)
    return {" & ".join(k): v for k, v in pairs.most_common()}


def near_duplicates(all_hash, hamming, max_examples=20):
    names, H, ds = [], [], []
    for d, m in all_hash.items():
        for f, h in m.items():
            names.append(f); H.append(h); ds.append(d)
    if not H:
        return {}
    H = np.unpackbits(np.stack(H), axis=1).astype(np.uint8)
    ds = np.array(ds)
    pair_counts, examples = Counter(), []
    step = max(1, int(2e8 // (len(H) * 64)))           # keep each distance block around 200 MB
    for i in range(0, len(H), step):
        dist = (H[i:i + step, None, :] != H[None, :, :]).sum(-1)
        ii, jj = np.nonzero(dist <= hamming)
        keep = jj > ii + i
        for a, b in zip(ii[keep] + i, jj[keep]):
            key = " & ".join(sorted({str(ds[a]), str(ds[b])})) if ds[a] != ds[b] else str(ds[a])
            pair_counts[key] += 1
            if ds[a] != ds[b] and len(examples) < max_examples:
                examples.append([f"{ds[a]}:{names[a]}", f"{ds[b]}:{names[b]}"])
    return {"hamming_threshold": hamming, "pairs_by_dataset": dict(pair_counts.most_common()), "cross_dataset_examples": examples}


# ---------------------------------------------------------------- task support (from what was found, nothing assumed)
def task_support(reports):
    by = {r["dataset"]: r for r in reports}

    def any_(pred):
        return [n for n, r in by.items() if pred(r)]
    healthy = any_(lambda r: "healthy" in r["dataset"].lower() and r["n_images"])
    caries = any_(lambda r: re.search(r"caries|carious", r["dataset"].lower()) and "fdi" not in r["dataset"].lower() and r["n_images"])
    yolo = any_(lambda r: r.get("yolo", {}).get("n_boxes"))
    coco = any_(lambda r: any(j.get("format") == "COCO" for j in r.get("json_files", [])))
    icdas = any_(lambda r: r.get("icdas_class_folders"))
    fdi = [n for n in yolo + coco if "fdi" in n.lower()]
    gvb = [n for n in yolo if "fdi" not in n.lower()]
    return {
        "healthy_vs_caries": {"datasets": healthy + caries, "label_source": "folder / archive name",
                              "supported": bool(healthy and caries)},
        "fdi_tooth_detection": {"datasets": fdi, "label_source": "box annotations (check class names = FDI numbers)",
                                "supported": bool(fdi)},
        "lesion_localization": {"datasets": gvb, "label_source": "YOLO boxes", "supported": bool(gvb)},
        "gv_black_classification": {"datasets": gvb, "label_source": "YOLO class ids (mapping must be verified, notebook §30)",
                                    "supported": bool(gvb)},
        "icdas_classification": {"datasets": icdas, "label_source": "ICDAS class folders", "supported": bool(icdas),
                                 "classes_present": sorted({k for n in icdas for k in by[n]["icdas_class_folders"]})},
        "patient_level_split_possible": any(r["patient_ids"]["found"] for r in reports),
    }


def to_markdown(out):
    L = ["# Dataset audit", "", "Generated by `backend/scripts/audit_datasets.py` (read-only).", ""]
    for r in out["datasets"]:
        L += [f"## {r['dataset']}", "", f"- files: {r['n_files']:,} · images: {r['n_images']:,} · extensions: {r['extensions']}"]
        if "n_unreadable" in r:
            L += [f"- unreadable images: {r['n_unreadable']} · formats: {r['formats']} · modes: {r['colour_modes']}",
                  f"- size (px): {r['size_px']}",
                  f"- exact duplicates: {r['exact_duplicate_groups']} groups, {r['exact_duplicate_extra_copies']} extra copies "
                  f"({r['exact_duplicates_across_folders']} groups span different folders)"]
        L.append(f"- images per folder: {r['images_per_folder'] if len(r['images_per_folder']) <= 40 else str(len(r['images_per_folder'])) + ' folders'}")
        if r.get("icdas_class_folders"):
            L.append(f"- ICDAS class folders: {r['icdas_class_folders']}")
        if r.get("yolo"):
            y = r["yolo"]
            L.append(f"- YOLO: {y['n_label_files_parsed_as_yolo']} label files, {y['n_boxes']} boxes, class ids {y['class_id_counts']}, "
                     f"names {y['class_names_file']}, malformed lines {y['malformed_lines']}, out-of-range boxes {y['boxes_out_of_range']}, "
                     f"images without labels {y['images_without_label_file']}, labels without images {y['label_files_without_image']}")
        for j in r.get("json_files", []):
            L.append(f"- JSON `{j['file']}`: " + json.dumps({k: v for k, v in j.items() if k != 'file'})[:400])
        for c in r.get("csv_files", []):
            L.append(f"- CSV `{c['file']}`: {c.get('n_rows')} rows, columns {c.get('columns')}")
        L += [f"- patient identifiers found: {r['patient_ids']['found']}", ""]
    L += ["## Across datasets", "", f"- identical files shared between datasets: {out['cross_dataset_exact_duplicates'] or 'none'}"]
    if out.get("near_duplicates"):
        L.append(f"- near-duplicates (dHash ≤ {out['near_duplicates']['hamming_threshold']}): {out['near_duplicates']['pairs_by_dataset']}")
    L += ["", "## Task support (from the files found)", ""]
    for k, v in out["task_support"].items():
        L.append(f"- **{k}**: {v}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="dataset ZIP files and/or folders")
    ap.add_argument("--no-decode", action="store_true", help="skip decoding images (structure + labels only; fast)")
    ap.add_argument("--near-dup", action="store_true", help="also compute dHash near-duplicates (slower)")
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    ap.add_argument("--md", default=None, help="write a Markdown summary")
    a = ap.parse_args(argv)

    reports, all_md5, all_hash = [], {}, {}
    for p in a.paths:
        print(f"Auditing {p} ...", flush=True)
        src = Source(p)
        rep, md5s, hashes = audit_source(src, check_images=not a.no_decode, want_hash=a.near_dup)
        reports.append(rep); all_md5[rep["dataset"]] = md5s; all_hash[rep["dataset"]] = hashes
        print(f"  {rep['n_images']:,} images, {rep.get('n_unreadable', 'n/a')} unreadable, "
              f"{rep.get('exact_duplicate_extra_copies', 'n/a')} exact duplicate copies")
    out = {"datasets": reports, "cross_dataset_exact_duplicates": cross_exact(all_md5)}
    if a.near_dup:
        out["near_duplicates"] = near_duplicates(all_hash, a.hamming)
    out["task_support"] = task_support(reports)
    print(json.dumps({"cross_dataset_exact_duplicates": out["cross_dataset_exact_duplicates"],
                      "task_support": out["task_support"]}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("JSON report:", a.out)
    if a.md:
        with open(a.md, "w") as f:
            f.write(to_markdown(out))
        print("Markdown summary:", a.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
