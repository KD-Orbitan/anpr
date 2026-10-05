import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(os.environ.get("ANPR_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def settings():
    config = read_json(ROOT / "configs/project.json")
    local = ROOT / "configs/project.local.json"
    if local.is_file():
        config.update(read_json(local))
    return config


def resolve(value):
    return (ROOT / value).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run(kind):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "runs" / (kind + "-" + stamp + "-" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def model_info(name):
    registry = read_json(ROOT / "models/registry.json")
    if name not in registry["models"]:
        raise ValueError("Unknown model: " + name)
    return registry["models"][name]
