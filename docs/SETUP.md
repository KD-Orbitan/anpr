# Setup and local assets

## Supported workflow

Run from a source checkout, using Python 3.9 for Paddle 2.6.2. Editable installation (`pip install -e .`) provides the CLI. A standalone wheel containing configs/vendor/assets is not supported. `ANPR_PROJECT_ROOT` can point to a source checkout when invoking the CLI elsewhere.

Use `requirements-ocr-cpu.txt` for cropped-image OCR, or `requirements-cpu.txt` for the full application. Model inference currently runs on CPU. For CPU training, install `requirements-training-cpu.txt`; it includes the application runtime and compatible training dependencies. Avoid installing the old vendor requirements directly because they request conflicting OpenCV packages. Do not install a newer PaddleOCR package over it.

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

## Open this folder in VS Code

Open `anpr_project` directly. Runtime code and datasets use paths inside this folder; sibling experiment folders are not required. In VS Code, run **Python: Select Interpreter** and select `.venv/Scripts/python.exe` on Windows. The environment is local and is not committed.

From a new PowerShell terminal at the project root:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training-cpu.txt
.\.venv\Scripts\python.exe scripts/doctor.py --mode training
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use `requirements-cpu.txt` instead for the app alone, or `requirements-ocr-cpu.txt` for OCR alone. Explicit interpreter paths avoid PowerShell activation-policy issues. Dependency pins target the historical Python 3.9 baseline; they are not a claim of support for current Python versions or a production internet-facing deployment.

`doctor.py --mode source` checks the checkout without third-party dependencies. `--mode ocr` imports the OCR runtime and verifies/loads the default model; `--mode app` also checks application dependencies and detector presence. `--mode training` checks imports and the default pretrained checkpoint; it does not validate a complete training run.

After providing authorized data and weights:

```powershell
.\.venv\Scripts\python.exe -m plateocr prepare-train --cpu --epochs 1
# Use the new run path printed above:
.\.venv\Scripts\python.exe scripts/smoke_train.py RUN
.\.venv\Scripts\python.exe scripts/generate_example.py
.\.venv\Scripts\python.exe -m plateocr infer data/examples/fictional_plate.png
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The smoke test uses a tiny subset to check training, validation, export and predictor reload. It creates separate run artifacts and does not replace the historical models. Real model quality needs a complete experiment and a permitted independent evaluation.

## API use

Start `python -m uvicorn api:app --host 127.0.0.1 --port 8000`, then open `http://127.0.0.1:8000/docs` for the interactive request/response schema. Both endpoints accept `{"image_base64": "<base64 encoded image bytes>"}`:

- `POST /ocr`: an already cropped plate; returns `plate` with text and confidence. Requires only OCR weights.
- `POST /anpr`: a vehicle image; returns a `plates` list. Requires OCR and detector weights.
- `GET /health`: service liveness and whether the ANPR pipeline has been loaded; it is not a model-readiness guarantee.

There is no authentication or production serving infrastructure in this research demo. Keep the default localhost binding for local use.
