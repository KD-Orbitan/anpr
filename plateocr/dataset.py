"""Create new group-separated manifests, preserving original images and labels."""
from collections import defaultdict
from datetime import datetime, timezone
import random
from pathlib import Path
from uuid import uuid4

from .data import load_splits
from .project import ROOT, resolve, settings, sha256, write_json


def group_split(records, seed=42):
    groups = defaultdict(list)
    for record in records:
        groups[record["label"]].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_train, n_val = int(len(keys) * 0.8), int(len(keys) * 0.1)
    if min(n_train, n_val, len(keys) - n_train - n_val) < 1:
        raise ValueError("Not enough distinct plate labels to create three non-empty splits")
    partitions = {"train": keys[:n_train], "val": keys[n_train:n_train + n_val], "test": keys[n_train + n_val:]}
    return {name: [r for key in values for r in groups[key]] for name, values in partitions.items()}


def prepare_data(seed=42):
    from .inference import read_image
    cfg = settings()
    chars = set(resolve(cfg["dictionary"]).read_text(encoding="utf-8-sig").splitlines())
    grouped, quarantine, duplicates = defaultdict(list), [], []
    source = load_splits(legacy=True)
    for split, records in source.items():
        for index, record in enumerate(records):
            if index % 1000 == 0:
                print(f"Checking {split}: {index}/{len(records)}", flush=True)
            item = dict(record, original_split=split)
            try:
                digest = sha256(record["image"])
                read_image(record["image"])
            except (OSError, ValueError) as error:
                quarantine.append(dict(item, reason=str(error)))
                continue
            # Group all decoded records before validating labels so conflicts cannot be hidden.
            grouped[digest].append(item)
    accepted = []
    for digest, records in sorted(grouped.items()):
        if len({r["label"] for r in records}) > 1:
            quarantine.extend(dict(r, sha256=digest, reason="identical image with conflicting labels") for r in records)
            continue
        if set(records[0]["label"]) - chars or len(records[0]["label"]) > cfg["max_text_length"]:
            quarantine.extend(dict(r, sha256=digest, reason="label outside dictionary or length limit") for r in records)
            continue
        accepted.append(records[0])
        duplicates.extend(dict(r, kept_image=records[0]["image"], sha256=digest) for r in records[1:])
    splits = group_split(accepted, seed)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = ROOT / "data/versions" / name
    folder.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for split, records in splits.items():
        path = folder / (split + ".txt")
        data_root = resolve(cfg["data_root"])
        path.write_text("".join(f'{Path(r["image"]).relative_to(data_root).as_posix()}\t{r["label"]}\n' for r in records), encoding="utf-8")
        manifests[split] = str(path.relative_to(ROOT))
    write_json(folder / "quarantine.json", quarantine)
    write_json(folder / "duplicates.json", duplicates)
    info = {"seed": seed, "group_by": "exact_plate_text_after_file_hash_deduplication",
            "manifests": manifests, "counts": {k: len(v) for k, v in splits.items()},
            "quarantined_records": len(quarantine), "duplicate_records": len(duplicates),
            "source_counts": {k: len(v) for k, v in source.items()},
            "source_manifest_sha256": {str(path): sha256(path) for path in sorted({r["source"] for records in source.values() for r in records})},
            "caveat": "Repartitioned historical data, not unseen data for existing checkpoints. Group by acquisition session/vehicle where available."}
    write_json(folder / "dataset.json", info)
    write_json(ROOT / "data/active.json", info)
    return info
