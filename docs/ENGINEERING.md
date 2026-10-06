# Engineering walkthrough

## Problem and scope

The project adapts a CRNN recognizer to Vietnamese license plate crops and combines it with an existing YOLO detector. The practical challenge is maintaining one consistent path from data preparation through training, export and application inference while retaining evidence of how each experiment was run.

The detector and neural network architectures are reused. Project work covers OCR fine-tuning, experiment reconstruction, data checks, preprocessing, evaluation and the application/tooling around these components. Historical artifacts do not establish a complete training-to-export lineage for every checkpoint; model names are identifiers, not verified accuracy claims.

## Design decisions and evidence

| Concern | Implementation | How it is checked |
| --- | --- | --- |
| Training/inference mismatch | BGR input, aspect-preserving resize, normalized padding, fixed `[3,48,256]` CRNN input | Preprocessing tests and train/export/reload smoke test |
| CTC decoding correctness | Remove consecutive repeats before removing blank tokens; check vocabulary size | Tests for repeats separated by blanks and incompatible dictionaries |
| Dataset leakage | Exact-file hashing, conflicting-label checks and grouping by plate text when preparing splits | Dataset audit and deterministic split tests |
| Experiment reproducibility | Snapshot config, manifests, dictionary, checkpoint hashes, engine hashes and environment in each run | Training launcher validates frozen inputs before execution |
| Misleading metrics | Full-plate exact match and CER; failed images remain in the denominator | Metric and failure-handling tests |
| Model provenance | Named registry entries with dictionary and weight hashes; new exports receive new names | Asset verification and registration checks |
| Integration | CLI, Streamlit, crop-only `/ocr`, detector-backed `/anpr` | API contract tests, local API requests and initial UI rendering |
| Publication scope | Explicit file inventory and separate code-only/full local bundles | Exclusion, checksum, path-boundary and branch-staging tests |

Exact hashing does not detect all near-duplicates. Grouping by plate text does not prove that all related scenes or vehicles are isolated. These checks reduce identifiable leakage; they are not a substitute for source-aware dataset review.

## Reproduce the workflow

1. Run the data-free tests from the README. This checks the code contract without private inputs.
2. Supply authorized source images and manifests, then run `prepare-data` and `audit --decode`.
3. Supply a compatible pretrained checkpoint and run `prepare-train`. Each new experiment gets a new run; do not edit frozen inputs afterward.
4. Run `scripts/smoke_train.py RUN` to verify a small CPU train/validation/export/reload cycle before a full experiment.
5. Train, evaluate on validation data and export a selected checkpoint. Register the export under a new name.
6. Freeze the model and preprocessing before a final independent test. Keep detector, OCR-on-crop and end-to-end metrics separate.

Commands and expected asset locations are in [setup](SETUP.md). The public release includes neither the pretrained checkpoint nor the fine-tuned weights, so steps 2–6 require additional authorized assets.

## What has been validated

The local Windows Python 3.9/Paddle 2.6.2 CPU environment has passed dependency checks, loading of historical OCR weights, both API routes with a fictional image, initial Streamlit rendering and a three-batch training/validation/export/reload smoke test. The exported graph uses `[N,3,48,256]` input and produces 34 output classes for the current dictionary plus CTC blank.

GitHub Actions runs data-free unit/integration tests and the public inventory check on Windows and Linux. It does not train a model, load private weights or evaluate the company dataset. See the live CI badge in the README for the current commit's status.

## Remaining research work

A complete new fine-tune and a permitted independent benchmark would provide new quality evidence. Useful follow-up experiments include reporting performance by image condition and plate layout, studying two-line crops, and measuring end-to-end errors separately from OCR errors. GPU training, full runtime installation on other platforms and production deployment are not validated by the current CPU smoke test.

Company evaluation results are withheld in this code release. The [dataset card](DATASET_CARD.md) records the outstanding provenance question around a historical evaluation configuration. This project does not claim that an unresolved historical evaluation was an untouched holdout.
