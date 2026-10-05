# Vietnamese License Plate OCR & ANPR

A computer vision project focused on fine-tuning a **CRNN license plate recognizer**, integrating it with an existing YOLO detector, and making experiments reproducible.

**Release scope:** source code, tests and documentation. Model weights and raw datasets are not bundled. Company-provided evaluation material and results are not published. Inference requires authorized weights; data-free tests can run immediately.

**Choose your publication scope:** [Hướng dẫn tiếng Việt: chỉ code hoặc code + OCR weights + số liệu tổng hợp](docs/CHOOSE_RELEASE_VI.md). Both previews can be built locally; nothing is pushed automatically.

```mermaid
flowchart LR
    A[Vehicle image] --> B[YOLO detector]
    B --> C[Plate crop]
    C --> D[BGR resize and normalization]
    D --> E[MobileNetV3 + BiLSTM]
    E --> F[CTC decoding]
    F --> G[Plate text and confidence]
```

## What I worked on

- Fine-tuning OCR with MobileNetV3, a bidirectional LSTM and CTC loss.
- Combining real and synthetic training sources and investigating difficult cases.
- Auditing duplicate images, conflicting labels, vocabulary coverage and split overlap.
- Building traceable training runs with frozen configs, dictionaries, manifests and hashes.
- Integrating detection and recognition through a CLI, Streamlit interface and FastAPI endpoint.
- Fixing preprocessing/export inconsistencies and final-batch handling in the local training engine.

YOLO, PaddleOCR and the neural network architectures are reused components. This project contributes the adaptation, experiments, data tooling and application integration.

## Quick start: source and tests

Use Python **3.9** for the Paddle 2.6.2 runtime. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python scripts/check_release.py
python -m plateocr --help
```

The test extra does not install Paddle or require model weights. The GitHub Actions workflow runs these tests; its remote status is only established after pushing and running the workflow.

For OCR inference, install `requirements-ocr-cpu.txt`. For YOLO, Streamlit and the API, install `requirements-cpu.txt` in a separate runtime environment (the test extra uses headless OpenCV). The complete CPU stack was freshly installed and tested on Windows with Python 3.9 and Paddle 2.6.2. Other platforms have not been validated for the full runtime. Run `python scripts/doctor.py --mode app` to check your setup.

## Models and demo

Obtain weights only from an authorized source. There is no approved public download in this release. See [setup](docs/SETUP.md) for expected files and verification:

```bash
python scripts/verify_assets.py
python scripts/generate_example.py
python -m plateocr infer data/examples/fictional_plate.png --model model92
python -m streamlit run app.py
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The generated image is a fictional input for demonstrating the workflow, not an accuracy benchmark. The application supports user-supplied plate crops and vehicle images; `--detect` enables YOLO before OCR. The API also offers `POST /ocr` for crops without detector weights. It accepts `POST /anpr` with `{"image_base64": "..."}` and returns all detected plates, text and separate detection/OCR confidence values.

## Training and evaluation

OCR uses **BGR, `[3, 48, 256]`**, aspect-preserving resize and padding. SVTR experiments using different image sizes are not part of this baseline.

Training data source supplied by the author: [Vietnamese License Plate OCR on Kaggle](https://www.kaggle.com/datasets/topkek69/vietnamese-license-plate-ocr/data). Download from the original source and follow its terms; the repository does not redistribute the dataset. Source licensing and the exact mapping of historical generated/augmented images still require documentation.

```bash
python -m plateocr prepare-data --seed 42
python -m plateocr audit --decode
python -m plateocr prepare-train --cpu --epochs 80 --batch-size 32
# Replace RUN with the path printed by prepare-train:
python -m plateocr train RUN
python -m plateocr eval RUN
python -m plateocr export RUN
python -m plateocr register-model my-crnn EXPORT_RUN
python -m plateocr evaluate --model my-crnn --manifest labels.txt --data-root images
```

See [setup](docs/SETUP.md), [dataset card](docs/DATASET_CARD.md) and [model card](docs/MODEL_CARD.md). Install `requirements-training-cpu.txt` for CPU training and run `python scripts/doctor.py --mode training` to check dependencies. `--pretrained` starts a new fine-tune; `--resume` restores optimizer/epoch state. Do not edit a prepared run's config; prepare a new run.

Evaluation reports full-plate exact-match accuracy and CER. Failed images remain in the denominator. OCR-on-crop performance is distinct from end-to-end ANPR performance. No company-test metrics are claimed in this public release while disclosure permission is unknown.

## Repository layout

```text
plateocr/             data, inference, evaluation, training, CLI
configs/             portable project defaults
models/              dictionary and model metadata; weights excluded
data/                local datasets and versioned manifests; excluded
runs/                local run snapshots, logs and checkpoints; excluded
tests/               unit and API contract tests
scripts/             export, asset verification and release tooling
vendor/PaddleOCR/    local engine, license and modification notes
docs/                setup, model/data cards and publication notes
app.py / api.py       Streamlit and FastAPI entry points
```

## Validation and limitations

The current suite passes **22 tests**, including API contracts, CTC decoding, portable manifests, dictionary integrity, publication-profile switching and release exclusions. Tests pass in both a minimal test environment and the fresh complete CPU environment on Windows. CI checks the source on Windows and Linux without private assets.

The complete CPU stack was installed into a new environment and passed dependency checks, existing-model loading, both API routes on a fictional image, initial Streamlit rendering, and a small train -> validation -> export -> reload integration test using public-source training data. Export uses the fixed `[N, 3, 48, 256]` input contract. These checks establish execution, not model quality. A complete new fine-tune and a new independent benchmark remain separate experiments; GPU training and full runtime installation on other platforms have not been verified.

Original-code licensing and weight redistribution permissions remain to be selected/confirmed. Third-party licenses remain applicable: see [notices](docs/THIRD_PARTY_NOTICES.md). Do not publish local datasets, predictions or reports when preparing a release; use the allowlisted export described in [publication](docs/PUBLICATION.md).
