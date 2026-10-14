# Vietnamese License Plate OCR — extended release

This source profile accompanies separately distributed OCR weights and aggregate evaluation results. Publish it only after confirming code, OCR-weight and aggregate-result permissions, plus applicable third-party terms.

See [aggregate results](docs/RESULTS.md), [setup](docs/SETUP.md), [model card](docs/MODEL_CARD.md) and [dataset card](docs/DATASET_CARD.md).

Download `ocr-models.zip` from this repository’s approved GitHub Release, extract into the repository root, then install `requirements-ocr-cpu.txt`. Run `python scripts/generate_example.py` and `python -m plateocr infer data/examples/fictional_plate.png --model model92`.

The OCR archive contains model92, model5plus and their dictionary. It does not contain YOLO weights, the original pretrained checkpoint, datasets, private labels or per-image predictions. Full vehicle-image detection requires a separately authorized detector.

Architecture: BGR input [3, 48, 256] → MobileNetV3 → BiLSTM → CTC. The project provides dataset auditing, traceable train/eval/export commands, CLI, Streamlit and FastAPI.

Run `python -m pip install -e ".[test]"` and `python -m unittest discover -s tests -v` for data-free tests. The synthetic example is not a quality benchmark.
