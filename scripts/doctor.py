"""Check a source checkout without downloading assets or changing configuration."""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plateocr.project import ROOT, settings, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('source', 'ocr', 'app', 'training'), default='source')
    args = parser.parse_args()
    failures = []
    print('Python:', sys.version.split()[0], '|', sys.executable)
    print('Project:', ROOT)
    if args.mode != 'source' and sys.version_info[:2] != (3, 9):
        failures.append('The documented runtime is Python 3.9; use a separate Python 3.9 environment.')
    modules = [] if args.mode == 'source' else ['numpy', 'cv2', 'paddle']
    if args.mode in ('app', 'training'):
        modules += ['torch', 'ultralytics', 'streamlit', 'fastapi', 'uvicorn']
    if args.mode == 'training':
        modules += ['shapely', 'skimage', 'imgaug', 'lmdb', 'visualdl', 'rapidfuzz', 'yaml']
    for name in modules:
        try:
            module = importlib.import_module(name)
            print('OK import', name, getattr(module, '__version__', ''))
        except Exception as error:
            failures.append(f'{name}: {error}')
    cfg = settings()
    for key in ('dictionary', 'paddleocr_root'):
        if not resolve(cfg[key]).exists():
            failures.append('Missing project asset: ' + key)
    if args.mode in ('ocr', 'app') and not failures:
        try:
            from plateocr.inference import Recognizer
            Recognizer()
            print('OK default OCR model: hashes and predictor loading')
        except Exception as error:
            failures.append('OCR model: ' + str(error))
    if args.mode == 'app' and not resolve(cfg['detector']).is_file():
        failures.append('Missing detector weights; see docs/SETUP.md')
    if args.mode == 'training':
        checkpoint = Path(str(resolve(cfg['pretrained_checkpoint'])) + '.pdparams')
        if not checkpoint.is_file():
            failures.append('Missing default pretrained checkpoint; supply --pretrained to prepare-train.')
        print('Use prepare-train and scripts/smoke_train.py to validate your data and training engine.')
    for failure in failures:
        print('FAIL:', failure)
    print('PASS' if not failures else f'{len(failures)} check(s) failed; see docs/SETUP.md')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
