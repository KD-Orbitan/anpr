# Local PaddleOCR engine

`PaddleOCR/ppocr`, `PaddleOCR/tools`, `LICENSE` and `requirements.txt` were copied from the existing `finetunegpu/PaddleOCR` project. The original source remains untouched. This is the local historical engine, not a claim to be an unmodified upstream release.

Project-specific patch: `tools/program.py` uses `len(dataloader)` on Windows as on other systems, in both training and evaluation. The legacy code subtracted one and silently skipped the final batch. The project uses `num_workers=0` and verifies full iteration with its training smoke test.

Export goes through `scripts/export_crnn.py`, explicitly setting CPU, dictionary class count, full checkpoint shape validation and input `[N, 3, 48, 256]`. It keeps Paddle's generated static code in the export run instead of the user profile.

Each prepared training run records SHA-256 for the engine's Python files. Retain the PaddleOCR license when distributing this source. Re-audit dependencies before upgrading the legacy Paddle runtime.
