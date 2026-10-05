# Data is not distributed with this repository

The author identifies [Vietnamese License Plate OCR on Kaggle](https://www.kaggle.com/datasets/topkek69/vietnamese-license-plate-ocr/data) as the training/validation source. Dataset version and redistribution terms still need verification; public accessibility alone is not a redistribution license.

The company-provided external evaluation dataset is private. Do not include its images, labels, filenames, predictions, hashes or per-image results in a public commit. It is not required to run the application with your own images.

Place authorized training data under `data/datasets/training/`. Each source manifest uses `relative/image.jpg<TAB>TEXT`, UTF-8. Source manifest filenames are configured in `configs/project.json`.

`prepare-data` creates versioned manifests under `data/versions/` and activates them with `data/active.json`. These remain local. Dataset paths are relative to the configured data root; frozen run manifests use absolute paths and must be regenerated after relocation.
