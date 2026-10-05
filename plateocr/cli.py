import argparse
import json
import sys
from pathlib import Path

from .project import ROOT, new_run, read_json, settings, write_json


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Vietnamese plate OCR: fixed CRNN input 3x48x256")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="Show model provenance and verified/unverified metrics")
    registration = sub.add_parser("register-model", help="Import a completed export under a new model name")
    registration.add_argument("name")
    registration.add_argument("export_run", type=Path)
    dataset = sub.add_parser("prepare-data", help="Build new deduplicated, label-grouped splits; preserve originals")
    dataset.add_argument("--seed", type=int, default=42)
    audit_parser = sub.add_parser("audit", help="Check labels, files and split leakage")
    audit_parser.add_argument("--decode", action="store_true")
    infer = sub.add_parser("infer", help="Read an image or a folder; crops by default")
    infer.add_argument("input", type=Path)
    infer.add_argument("--model", default=settings()["default_model"])
    infer.add_argument("--detect", action="store_true", help="Run YOLO before OCR; return all detected plates")
    evaluate = sub.add_parser("evaluate", help="Evaluate OCR on a labelled crop manifest")
    evaluate.add_argument("--model", default=settings()["default_model"])
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--color-order", choices=["BGR", "RGB"], default="BGR")
    evaluate.add_argument("--legacy-preprocess", action="store_true", help="Use historical min_width=32; combine with RGB to reproduce demo")
    evaluate.add_argument("--limit", type=int, help="Smoke test only; omitted means full manifest")
    prepare = sub.add_parser("prepare-train", help="Audit data and freeze a training configuration; does not train")
    prepare.add_argument("--epochs", type=int, default=80)
    prepare.add_argument("--learning-rate", type=float, default=0.001)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--cpu", action="store_true")
    group = prepare.add_mutually_exclusive_group()
    group.add_argument("--pretrained", type=Path)
    group.add_argument("--resume", type=Path)
    for action in ("train", "eval", "export"):
        action_parser = sub.add_parser(action, help="Run PaddleOCR " + action + " using a prepared run")
        action_parser.add_argument("run", type=Path)
        if action != "train":
            action_parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "models":
            print(json.dumps(read_json(ROOT / "models/registry.json"), ensure_ascii=False, indent=2))
        elif args.command == "register-model":
            from .registry import register
            print(json.dumps(register(args.name, args.export_run), ensure_ascii=False, indent=2))
        elif args.command == "prepare-data":
            from .dataset import prepare_data
            print(json.dumps(prepare_data(args.seed), ensure_ascii=False, indent=2))
        elif args.command == "audit":
            from .data import audit
            report = audit(args.decode)
            write_json(ROOT / "reports/data_audit.json", report)
            print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False, indent=2))
            if report["errors"]:
                raise SystemExit(2)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            run, summary = evaluate(args.model, args.manifest.resolve(), args.data_root.resolve(),
                                    args.color_order, args.legacy_preprocess, args.limit)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print("Report:", run)
            if summary["failed_images"]:
                raise SystemExit(2)
        elif args.command == "infer":
            from .inference import Pipeline, Recognizer, read_image
            files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) if args.input.is_dir() else [args.input]
            if not files:
                raise ValueError("No images found")
            engine = Pipeline(args.model) if args.detect else Recognizer(args.model)
            rows = []
            for path in files:
                rows.append({"image": str(path.resolve()), "result": engine.predict(read_image(path))})
            run = new_run("infer")
            write_json(run / "predictions.json", {"model": args.model, "detect": args.detect, "images": rows})
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            print("Report:", run)
        elif args.command == "prepare-train":
            from .training import prepare
            print(prepare(args.epochs, args.learning_rate, args.batch_size, args.cpu, args.pretrained, args.resume))
        else:
            from .training import launch
            print(launch(args.run, args.command, getattr(args, "checkpoint", None)))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")
