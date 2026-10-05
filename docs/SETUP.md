# Setup and local assets

## Supported workflow

Run from a source checkout, using Python 3.9 for Paddle 2.5. Editable installation (`pip install -e .`) provides the CLI. A standalone wheel containing configs/vendor/assets is not supported. `ANPR_PROJECT_ROOT` can point to a source checkout when invoking the CLI elsewhere.

Use `requirements-ocr-cpu.txt` for cropped-image OCR, or `requirements-cpu.txt` for the full application. Model inference currently runs on CPU. For training, install the local engine's required dependencies from `vendor/PaddleOCR/requirements.txt` alongside the compatible runtime; this is a historical baseline, not a fresh-install training lockfile. Do not install a newer PaddleOCR package over it.

## Authorized weights

Expected locations:

```text
models/detector/model_detect_m.pt
models/model92/inference.pdmodel
models/model92/inference.pdiparams
models/model5plus/inference.pdmodel
models/model5plus/inference.pdiparams
models/pretrained/crnn_base.pdparams
```

The last file is for training. Inference uses the selected OCR model and, for full ANPR, the detector. `scripts/verify_assets.py` compares the supplied historical inference files against `models/registry.json`. There is no public model URL while redistribution permission remains unknown. Do not substitute arbitrary binaries under the historical names.

After training/export, `register-model NEW_NAME EXPORT_RUN` copies the export under a new name and records provenance and hashes. Select it with `--model NEW_NAME`; use the project configuration to change the app default.

## Dataset manifests

Store authorized training sources under `data/datasets/training/`. Default source manifest names are `gen_train.txt`, `real_train.txt`, `gen_val.txt`, `real_val.txt`, `gen_test.txt`, `real_test.txt`. These describe the historical local layout, not a guarantee that the Kaggle download has those exact filenames.

Each line contains `relative/image.jpg<TAB>LABEL`. Edit `splits` in `configs/project.json` to match your manifests, or create an ignored `configs/project.local.json` with overrides. The source dataset needs at least ten distinct valid plate labels for the default grouped split. The vocabulary is fixed by `models/characters.txt`; changing it requires a compatible model head and dictionary registration.

`prepare-data` groups public-source records by plate text and creates local versioned manifests. Do not put the independent company test set into those source manifests. Evaluate any authorized external test separately, after freezing the model and preprocessing. Reports stay local.

## Portable local project

Paths in committed defaults remain within the project. Prepared training runs intentionally freeze absolute paths; regenerate them after moving the checkout. Historical local reports and configs are not portable release inputs and are excluded from the public export.
