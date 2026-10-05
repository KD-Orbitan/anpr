import csv
import time

from .data import read_manifest
from .project import new_run, sha256, write_json


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        nxt = [i]
        for j, other in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (char != other)))
        row = nxt
    return row[-1]


def metrics(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    correct = sum(not r.get("error") and r["prediction"] == r["label"] for r in rows)
    edits = sum(edit_distance(r["prediction"], r["label"]) for r in rows)
    chars = sum(len(r["label"]) for r in rows)
    return {"samples": len(rows), "correct": correct, "exact_accuracy": correct / len(rows),
            "cer": edits / max(chars, 1), "failed_images": sum(bool(r.get("error")) for r in rows)}


def evaluate(model, manifest, data_root, color_order="BGR", legacy_preprocess=False, limit=None):
    from .inference import Recognizer, read_image
    records = read_manifest(manifest, data_root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        records = records[:limit]
    recognizer = Recognizer(model, color_order, legacy_preprocess)
    run = new_run("eval-" + model)
    rows = []
    started = time.perf_counter()
    for record in records:
        row = {"image": record["image"], "label": record["label"], "prediction": "", "confidence": 0.0, "error": ""}
        try:
            result = recognizer.predict(read_image(record["image"]))
            row.update(prediction=result["text"], confidence=result["confidence"])
        except (OSError, ValueError, RuntimeError) as error:
            row["error"] = str(error)
        rows.append(row)
    with (run / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {**metrics(rows), "model": model, "model_info": recognizer.info,
               "manifest": str(manifest), "manifest_sha256": sha256(manifest),
               "color_order": color_order, "min_width": recognizer.min_width,
               "limit": limit, "elapsed_seconds": time.perf_counter() - started,
               "scope": "OCR on crops; historical splits may have been used in training/model selection."}
    write_json(run / "metrics.json", summary)
    return run, summary
