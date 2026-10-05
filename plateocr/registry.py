import re
from pathlib import Path
import shutil

from .project import ROOT, read_json, settings, resolve, sha256, write_json


def register(name, export_run):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("Model name must contain only letters, digits, underscores or hyphens")
    path = ROOT / "models/registry.json"
    registry = read_json(path)
    if name in registry["models"] or (ROOT / "models" / name).exists():
        raise ValueError("Model name already exists; use a new name")
    export_run = Path(export_run).resolve()
    info = read_json(export_run / "run.json")
    if info.get("status") != "completed" or info.get("action") != "export":
        raise ValueError("Expected a completed export run")
    source = export_run / "model"
    if (source / "characters.txt").read_text(encoding="utf-8-sig").splitlines() != resolve(settings()["dictionary"]).read_text(encoding="utf-8-sig").splitlines():
        raise ValueError("Export dictionary does not match project dictionary")
    files = [source / "inference.pdmodel", source / "inference.pdiparams"]
    checksums = {p.name: sha256(p) for p in files}
    target = ROOT / "models" / name
    target.mkdir()
    for file in files:
        shutil.copy2(file, target / file.name)
    registry["models"][name] = {"path": str(target.relative_to(ROOT)), "sha256": checksums,
                                "image_shape": [3, 48, 256], "export_run": str(export_run),
                                "training_run": info["parent_training_run"],
                                "checkpoint_sha256": info["checkpoint_sha256"],
                                "historical_accuracy": {"value": None, "status": "not_evaluated", "dataset": None}}
    write_json(path, registry)
    return registry["models"][name]
