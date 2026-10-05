"""Explicit public-file inventory; does not rely on .gitignore for ZIP safety."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOP = {'README.md', '.gitignore', '.gitattributes', 'pyproject.toml', 'requirements-cpu.txt',
       'requirements-ocr-cpu.txt', 'api.py', 'app.py'}
DOCS = {'SETUP.md', 'MODEL_CARD.md', 'DATASET_CARD.md', 'THIRD_PARTY_NOTICES.md', 'PUBLICATION.md'}
SCRIPTS = {'check_release.py', 'export_public.py', 'verify_assets.py', 'export_crnn.py',
           'smoke_train.py', 'generate_example.py'}


def allowed(path):
    parts = path.parts
    name = path.as_posix()
    if '__pycache__' in parts or path.suffix == '.pyc':
        return False
    if len(parts) == 1:
        return name in TOP
    if parts[0] in ('plateocr', 'tests'):
        return path.suffix == '.py'
    if parts[0] == 'docs':
        return len(parts) == 2 and parts[1] in DOCS
    if parts[0] == 'scripts':
        return len(parts) == 2 and parts[1] in SCRIPTS
    if name in ('configs/project.json', 'models/registry.json', 'models/characters.txt',
                'data/README.md', 'reports/README.md', '.github/workflows/tests.yml', 'vendor/README.md'):
        return True
    if parts[:2] == ('vendor', 'PaddleOCR'):
        return name in ('vendor/PaddleOCR/LICENSE', 'vendor/PaddleOCR/requirements.txt') or (
            len(parts) > 3 and parts[2] in ('ppocr', 'tools') and path.suffix == '.py')
    return False


def files(root=ROOT):
    return sorted(path for path in root.rglob('*') if path.is_file() and allowed(path.relative_to(root)))


def check(root=ROOT):
    errors = []
    for path in files(root):
        relative = path.relative_to(root)
        if relative.parts[0] == 'vendor':
            continue
        text = path.read_text(encoding='utf-8-sig')
        if re.search(r'[A-Za-z]:[\\/](?:AI|Users)', text):
            errors.append(str(relative) + ': machine-specific absolute path')
    cfg = __import__('json').loads((root / 'configs/project.json').read_text(encoding='utf-8'))
    for key in ('data_root', 'paddleocr_root', 'pretrained_checkpoint', 'dictionary', 'detector'):
        value = Path(cfg[key])
        if value.is_absolute() or '..' in value.parts:
            errors.append('External project dependency: ' + key)
    # If this is a Git checkout, report tracked files outside the public inventory.
    if (root / '.git').exists():
        result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, check=True)
        for value in result.stdout.decode('utf-8').split('\0'):
            if value and not allowed(Path(value)):
                errors.append('Non-public tracked file: ' + value)
    return errors


if __name__ == '__main__':
    issues = check()
    print('\n'.join(issues) if issues else f'Public inventory passed: {len(files())} files. Raw data, weights and reports excluded.')
    raise SystemExit(bool(issues))
