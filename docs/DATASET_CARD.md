# Dataset card

## Training and validation

The author identifies [Vietnamese License Plate OCR by topkek69 on Kaggle](https://www.kaggle.com/datasets/topkek69/vietnamese-license-plate-ocr/data) as the public training/validation source. The page's licensing terms have not been verified here. The exact dataset version and mapping of historical generated/augmented images to source files still need documentation. Raw data is not redistributed by this repository.

Data preparation checks readable images, character vocabulary, conflicting labels and duplicate file hashes. A new split groups identical plate text together and uses seed 42. Exact byte hashes do not identify every near-duplicate; capture-session or vehicle metadata should be used when available.

## Private external evaluation

The author identifies the company-provided dataset as an independent final test set, separate from public training/validation data. Its contents and per-image evaluation outputs are confidential and excluded from distribution.

A historical configuration references part of this dataset in its Eval section. This does not establish that its images were used for gradient training. However, whether that run influenced checkpoint/model selection must be reconciled before claiming a strictly untouched holdout for every historical model. The public repository does not make that blanket claim.

No private labels, sample filenames or image hashes are needed for installation, unit tests or inference on user-provided images. Aggregate company-dataset metrics are withheld pending permission.
