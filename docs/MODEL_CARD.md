# Model card

## Task and architecture

Read Vietnamese license plates from cropped images. The application can first locate plates using an existing YOLO detector.

- OCR backbone: MobileNetV3, scale 0.5.
- Sequence model: bidirectional LSTM, hidden size 96.
- Objective and decoder: CTC, blank index 0.
- Input: BGR, `[N, 3, 48, 256]`, aspect-preserving resize and zero padding after normalization to [-1, 1].
- Vocabulary: `models/characters.txt`; the class order must not change.

## Project contributions

OCR fine-tuning and experimentation, data preparation and duplicate/conflicting-label checks, evaluation tooling, traceable run snapshots, and YOLO–OCR integration with CLI, Streamlit and FastAPI. The detector and PaddleOCR architecture are reused components, not architectures developed from scratch by this project.

## Checkpoints and distribution

The local project has historical OCR checkpoints named `model92` and `model5plus`. Model names are identifiers, not quality guarantees. Historical export-to-training-run associations are not fully established.

Weights are not bundled or downloaded automatically. Redistribution permissions and model licenses must be confirmed before providing public downloads. A user with authorized weights can place them in the documented model directories and verify their hashes.

## Evaluation and limitations

Report full-plate exact match and character error rate (CER) separately from detector or end-to-end ANPR metrics. Confidence is not accuracy. Training smoke-test results only verify execution and are not quality estimates.

Company-data metrics are withheld pending permission. See the dataset card for the author's independent-test description and an outstanding historical Eval-reference question. Blurred, occluded, tilted or unfamiliar plate layouts can produce errors; the current vocabulary does not cover every possible identifier.
