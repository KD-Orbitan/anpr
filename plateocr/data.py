"""Read and audit existing manifests without changing historical datasets."""
from collections import Counter, defaultdict
from pathlib import Path

from .project import ROOT, read_json, resolve, settings, sha256


def read_manifest(path, data_root):
    records = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"{path}:{number}: expected image<TAB>label")
        name, label = parts
        image = (Path(data_root) / name.replace("\\", "/")).resolve()
        key = str(image).casefold()
        if key in seen:
            raise ValueError(f"{path}:{number}: duplicate image: {name}")
        seen.add(key)
        records.append({"image": str(image), "label": label, "source": str(path), "line": number})
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def load_splits(legacy=False):
    cfg = settings()
    root = resolve(cfg["data_root"])
    active = ROOT / "data/active.json"
    if not legacy and active.is_file():
        dataset = read_json(active)
        return {split: read_manifest(resolve(path), root) for split, path in dataset["manifests"].items()}
    return {split: [record for name in names for record in read_manifest(root / name, root)]
            for split, names in cfg["splits"].items()}


def audit(check_images=False):
    cfg = settings()
    chars = set(resolve(cfg["dictionary"]).read_text(encoding="utf-8-sig").splitlines())
    splits = load_splits()
    errors, warnings = [], []
    locations, contents, labels = defaultdict(list), defaultdict(list), defaultdict(set)
    counts = {}
    for split, records in splits.items():
        counts[split] = len(records)
        for record in records:
            path, label = Path(record["image"]), record["label"]
            locations[str(path).casefold()].append(split)
            labels[label].add(split)
            if len(label) > cfg["max_text_length"] or set(label) - chars:
                errors.append({"kind": "invalid_label", "split": split, **record})
            if not path.is_file():
                errors.append({"kind": "missing_image", "split": split, **record})
                continue
            contents[sha256(path)].append((split, str(path), label))
            if check_images:
                import cv2
                if cv2.imread(str(path)) is None:
                    errors.append({"kind": "unreadable_image", "split": split, **record})
    for path, memberships in locations.items():
        if len(memberships) > 1:
            errors.append({"kind": "repeated_path", "image": path, "splits": memberships})
    for digest, entries in contents.items():
        if len({entry[0] for entry in entries}) > 1:
            errors.append({"kind": "cross_split_identical_file", "sha256": digest, "entries": entries})
        if len({entry[2] for entry in entries}) > 1:
            errors.append({"kind": "conflicting_labels", "sha256": digest, "entries": entries})
    shared = sum(len(value) > 1 for value in labels.values())
    if shared:
        warnings.append(f"{shared} plate texts occur in multiple splits; review vehicle/source grouping.")
    return {"counts": counts, "errors": errors, "warnings": warnings,
            "error_counts": dict(Counter(item["kind"] for item in errors)),
            "checked_image_decode": check_images,
            "scope": "Exact file hashes; does not detect all near-duplicate images."}


def export_manifests(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, records in load_splits().items():
        path = directory / (split + ".txt")
        path.write_text("".join(f'{r["image"]}\t{r["label"]}\n' for r in records), encoding="utf-8")
        result[split] = str(path)
    return result
