"""Small train/eval/export smoke test; never modifies a prepared run."""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plateocr.project import new_run, read_json, sha256, write_json
from plateocr.training import launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_run", type=Path)
    args = parser.parse_args()
    source = args.prepared_run.resolve()
    config = copy.deepcopy(read_json(source / "config.yml"))
    metadata = read_json(source / "run.json")
    if config["Global"]["checkpoints"]:
        raise ValueError("Use a fresh pretrained run, not resume, for smoke testing")
    target = new_run("smoke-train")
    (target / "manifests").mkdir()
    (target / "characters.txt").write_bytes((source / "characters.txt").read_bytes())
    for split in ("train", "val", "test"):
        lines = (source / "manifests" / (split + ".txt")).read_text(encoding="utf-8").splitlines()[:12]
        (target / "manifests" / (split + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    config["Global"].update(epoch_num=1, use_gpu=False, save_model_dir=str(target / "checkpoints"),
                            character_dict_path=str(target / "characters.txt"), eval_batch_step=[0, 1],
                            save_epoch_step=1, print_batch_step=1, save_res_path=str(target / "predictions.txt"))
    config["PostProcess"]["character_dict_path"] = str(target / "characters.txt")
    for section, split in (("Train", "train"), ("Eval", "val")):
        config[section]["dataset"]["label_file_list"] = [str(target / "manifests" / (split + ".txt"))]
        config[section]["loader"]["batch_size_per_card"] = 4
    write_json(target / "config.yml", config)
    metadata.update(status="prepared", purpose="small integration test, not a production model",
                    config_sha256=sha256(target / "config.yml"),
                    manifest_sha256={s: sha256(target / "manifests" / (s + ".txt")) for s in ("train", "val", "test")})
    write_json(target / "run.json", metadata)
    launch(target, "train")
    log = (target / "console.log").read_text(encoding="utf-8")
    if "global_step:" not in log or "cur metric" not in log:
        raise RuntimeError("Smoke test did not train and evaluate a batch")
    exported = launch(target, "export", target / "checkpoints/latest")
    import numpy as np
    from paddle.inference import Config, create_predictor
    from plateocr.inference import decode_ctc, preprocess, read_image
    predictor_config = Config(str(exported / "model/inference.pdmodel"), str(exported / "model/inference.pdiparams"))
    predictor_config.disable_gpu()
    predictor_config.disable_glog_info()
    predictor_config.set_cpu_math_library_num_threads(2)
    predictor = create_predictor(predictor_config)
    sample = (target / "manifests/val.txt").read_text(encoding="utf-8").splitlines()[0].split("\t")[0]
    predictor.get_input_handle(predictor.get_input_names()[0]).copy_from_cpu(preprocess(read_image(sample)))
    predictor.run()
    output = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    assert np.isfinite(output).all(), "Non-finite exported model output"
    characters = (target / "characters.txt").read_text(encoding="utf-8").splitlines()
    decode_ctc(output, characters)
    write_json(target / "verification.json", {"training_updates": "global_step: 3" in log,
               "validation_ran": "cur metric" in log, "exported_model_loaded": True,
               "input_shape": [1, 3, 48, 256], "output_shape": list(output.shape),
               "export_run": str(exported), "note": "Smoke test only, not model quality evaluation"})
    print("SMOKE_TRAIN_RUN:", target)
    print("SMOKE_EXPORT_RUN:", exported)


if __name__ == "__main__":
    main()
