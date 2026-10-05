"""Generate immutable run inputs and delegate training to the local PaddleOCR."""
import copy
import os
from pathlib import Path
import subprocess
import sys

from .data import audit, export_manifests
from .project import ROOT, new_run, read_json, resolve, settings, sha256, write_json


def prepare(epochs=80, learning_rate=0.001, batch_size=32, cpu=False, pretrained=None, resume=None):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if pretrained and resume:
        raise ValueError("Choose pretrained OR resume, not both")
    report = audit(check_images=True)
    if report["errors"]:
        write_json(ROOT / "reports/data_audit.json", report)
        raise ValueError("Dataset audit failed; see reports/data_audit.json. No training started.")
    cfg = settings()
    engine = resolve(cfg["paddleocr_root"])
    if cfg["image_shape"] != [3, 48, 256] or cfg["color_order"] != "BGR":
        raise ValueError("Training requires CRNN input [3, 48, 256], BGR")
    default = resolve(cfg["pretrained_checkpoint"])
    checkpoint = Path(resume or pretrained or default).resolve()
    if checkpoint.suffix == ".pdparams":
        checkpoint = checkpoint.with_suffix("")
    if not Path(str(checkpoint) + ".pdparams").is_file():
        raise FileNotFoundError(str(checkpoint) + ".pdparams")
    if resume:
        for suffix in (".pdopt", ".states"):
            if not Path(str(checkpoint) + suffix).is_file():
                raise FileNotFoundError("Resume requires " + str(checkpoint) + suffix)
    run = new_run("train")
    manifests = export_manifests(run / "manifests")
    dictionary = run / "characters.txt"
    dictionary.write_bytes(resolve(cfg["dictionary"]).read_bytes())
    encoding = {"CTCLabelEncode": {"max_text_length": 10, "use_space_char": False}}
    resize = {"RecResizeImg": {"image_shape": [3, 48, 256], "padding": True}}
    keep = {"KeepKeys": {"keep_keys": ["image", "label", "length"]}}

    def dataset(split):
        transforms = [{"DecodeImage": {"img_mode": "BGR", "channel_first": False}}, copy.deepcopy(encoding)]
        if split == "train":
            transforms.append({"RecAug": {"use_tia": True, "aug_prob": 0.4}})
        transforms += [copy.deepcopy(resize), copy.deepcopy(keep)]
        return {"dataset": {"name": "SimpleDataSet", "data_dir": str(run),
                            "label_file_list": [manifests[split]], "transforms": transforms},
                "loader": {"shuffle": split == "train", "drop_last": split == "train",
                           "batch_size_per_card": batch_size, "num_workers": 0, "use_shared_memory": False}}

    config = {
        "Global": {"use_gpu": not cpu, "use_amp": False, "epoch_num": epochs,
                   "seed": 42, "log_smooth_window": 20, "print_batch_step": 20,
                   "save_model_dir": str(run / "checkpoints"), "save_epoch_step": 5,
                   "eval_batch_step": [0, 100], "cal_metric_during_train": True,
                   "pretrained_model": None if resume else str(checkpoint),
                   "checkpoints": str(checkpoint) if resume else None,
                   "character_dict_path": str(dictionary), "max_text_length": 10,
                   "use_space_char": False, "use_visualdl": False,
                   "save_res_path": str(run / "predictions.txt")},
        "Optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999,
                      "lr": {"name": "Cosine", "learning_rate": learning_rate, "warmup_epoch": 0},
                      "regularizer": {"name": "L2", "factor": 0.0001}},
        "Architecture": {"model_type": "rec", "algorithm": "CRNN", "Transform": None,
                         "Backbone": {"name": "MobileNetV3", "scale": 0.5, "model_name": "large"},
                         "Neck": {"name": "SequenceEncoder", "encoder_type": "rnn", "hidden_size": 96},
                         "Head": {"name": "CTCHead", "fc_decay": 0.00001}},
        "Loss": {"name": "CTCLoss"},
        "PostProcess": {"name": "CTCLabelDecode", "character_dict_path": str(dictionary), "use_space_char": False},
        "Metric": {"name": "RecMetric", "main_indicator": "acc", "ignore_space": False, "is_filter": False},
        "Train": dataset("train"), "Eval": dataset("val")}
    # JSON is valid YAML; avoids a configuration-generation dependency.
    write_json(run / "config.yml", config)
    write_json(run / "data_audit.json", report)
    write_json(run / "run.json", {"status": "prepared", "python": sys.executable,
               "python_version": sys.version, "engine": str(engine), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256(str(checkpoint) + ".pdparams"),
               "dictionary_sha256": sha256(dictionary),
               "manifest_sha256": {k: sha256(v) for k, v in manifests.items()},
               "config_sha256": sha256(run / "config.yml"),
               "engine_files_sha256": {str(p.relative_to(engine)): sha256(p)
                                       for folder in ("ppocr", "tools")
                                       for p in sorted((engine / folder).rglob("*.py"))}})
    return run


def launch(run, action="train", checkpoint=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run = Path(run).resolve()
    metadata = read_json(run / "run.json")
    original = run / "config.yml"
    if sha256(original) != metadata["config_sha256"]:
        raise ValueError("Prepared config changed. Prepare a new run instead of editing an existing run.")
    if sha256(run / "characters.txt") != metadata["dictionary_sha256"]:
        raise ValueError("Dictionary changed")
    engine = Path(metadata["engine"])
    for relative, digest in metadata["engine_files_sha256"].items():
        if sha256(engine / relative) != digest:
            raise ValueError("PaddleOCR source changed since prepare: " + relative)
    config = read_json(original)
    if action == "train":
        if metadata["status"] != "prepared":
            raise ValueError("Run already started; prepare a new run with --resume to continue.")
        for split, digest in metadata["manifest_sha256"].items():
            if sha256(run / "manifests" / (split + ".txt")) != digest:
                raise ValueError("Manifest changed: " + split)
        if sha256(metadata["checkpoint"] + ".pdparams") != metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        target = run
    else:
        checkpoint = Path(checkpoint or run / "checkpoints/best_accuracy").resolve()
        if checkpoint.suffix == ".pdparams":
            checkpoint = checkpoint.with_suffix("")
        if not Path(str(checkpoint) + ".pdparams").is_file():
            raise FileNotFoundError(str(checkpoint) + ".pdparams")
        target = new_run(action)
        config["Global"].update(pretrained_model=str(checkpoint), checkpoints=None, use_gpu=False)
        config["Global"]["save_inference_dir"] = str(target / "model")
        config["Global"]["save_model_dir"] = str(target / "logs")
        write_json(target / "config.yml", config)
    command = [sys.executable, str(engine / "tools" / (action + ".py")), "-c", str(target / "config.yml")]
    if action == "export":
        command = [sys.executable, str(ROOT / "scripts/export_crnn.py"), "--engine", str(engine),
                   "-c", str(target / "config.yml")]
    env = dict(os.environ, PYTHONUTF8="1")
    with (target / "environment.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    state = dict(metadata, status="running", action=action, command=command)
    state["config_sha256"] = sha256(target / "config.yml")
    if action != "train":
        state.update(parent_training_run=str(run), checkpoint=str(checkpoint),
                     checkpoint_sha256=sha256(str(checkpoint) + ".pdparams"))
    write_json(target / "run.json", state)
    try:
        with (target / "console.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=engine, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="")
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        state.update(status="completed" if code == 0 else "failed", exit_code=code)
        write_json(target / "run.json", state)
        if code:
            raise RuntimeError(f"PaddleOCR exited with code {code}; see {target / 'console.log'}")
    except BaseException:
        if state["status"] == "running":
            state["status"] = "interrupted"
            write_json(target / "run.json", state)
        raise
    return target
