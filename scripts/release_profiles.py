"""Build two local publication bundles; stage only with explicit permission confirmation.

No command pushes Git or uploads assets. Private evidence never enters source snapshots.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
import zipfile

from check_release import ROOT, allowed, check, files


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def within(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or path.is_symlink():
        raise ValueError('Path must stay inside its root: ' + str(relative))
    return path


def summaries(root, input_path, registry):
    """Project fixed numeric fields, not arbitrary strings/paths from private reports."""
    inputs = read(input_path)
    output = []
    for report_name in inputs['metric_reports']:
        row = read(within(root, report_name))
        model = row['model']
        if model not in ('model92', 'model5plus') or model not in registry['models']:
            raise ValueError('Unrecognized historical OCR model')
        if row.get('limit') is not None:
            raise ValueError('Subset/smoke metrics cannot be published as full evaluation')
        count, correct, failures = row['samples'], row['correct'], row['failed_images']
        if any(type(v) is not int for v in (count, correct, failures)) or not 0 <= correct <= count or count < 1 or not 0 <= failures <= count:
            raise ValueError('Invalid sample counts')
        acc, cer = row['exact_accuracy'], row['cer']
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (acc, cer)) or cer < 0 or abs(acc - correct / count) > 1e-8:
            raise ValueError('Invalid aggregate metrics')
        if row['color_order'] not in ('BGR', 'RGB') or row['min_width'] not in (1, 32):
            raise ValueError('Unsupported preprocessing')
        registered = registry['models'][model]['sha256']
        if row['model_info']['sha256']['inference.pdiparams'] != registered['inference.pdiparams']:
            raise ValueError('Metric/checkpoint hash mismatch')
        output.append({'model': model, 'dataset': 'private_external_test', 'samples': count,
                       'correct': correct, 'exact_accuracy': acc, 'cer': cer, 'failed_images': failures,
                       'color_order': row['color_order'], 'min_width': row['min_width']})
    if not output:
        raise ValueError('No selected aggregate reports')
    return {'scope': 'OCR on plate crops; not end-to-end ANPR',
            'protocol_note': 'Author reports external final-test data. Historical checkpoint-selection provenance remains under review.',
            'evaluations': output}


def build(profile, root=ROOT):
    if profile not in ('code-only', 'full'):
        raise ValueError('Unknown profile')
    issues = check(root)
    if issues:
        raise ValueError('\n'.join(issues))
    registry = read(root / 'models/registry.json')
    doc_names = ('README.md', 'docs/SETUP.md', 'docs/MODEL_CARD.md', 'docs/DATASET_CARD.md')
    templates = root / 'configs/code-only-docs.json'
    original_docs = read(templates) if templates.exists() else {name: (root / name).read_text(encoding='utf-8') for name in doc_names}
    if set(original_docs) != set(doc_names) or not all(isinstance(value, str) for value in original_docs.values()):
        raise ValueError('Invalid code-only documentation templates')
    summary, assets = None, []
    if profile == 'full':
        summary = summaries(root, root / '.private/release-inputs.json', registry)
        for model in ('model92', 'model5plus'):
            entry = registry['models'][model]
            for name in ('inference.pdmodel', 'inference.pdiparams'):
                path = within(root, entry['path'] + '/' + name)
                if digest(path) != entry['sha256'][name]:
                    raise ValueError('Weight hash mismatch: ' + str(path))
                assets.append((path, f'models/{model}/{name}'))
        dictionary = root / 'models/characters.txt'
        if digest(dictionary) != registry['dictionary_sha256']:
            raise ValueError('Dictionary hash mismatch')
        assets.append((dictionary, 'models/characters.txt'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
    bundle = root / '.private/releases' / (stamp + '-' + profile)
    source = bundle / 'source'
    source.mkdir(parents=True, exist_ok=False)
    for path in files(root):
        relative = path.relative_to(root)
        # Building code-only from a previously full source cannot retain its results page.
        if not allowed(relative, 'code-only') or relative.as_posix() == 'publication.json':
            continue
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(source / 'publication.json', {'schema_version': 1, 'profile': profile})
    for name, text in original_docs.items():
        (source / name).write_text(text, encoding='utf-8')
    if profile == 'full':
        write(source / 'configs/code-only-docs.json', original_docs)
        asset_dir = bundle / 'assets'
        asset_dir.mkdir()
        with zipfile.ZipFile(asset_dir / 'ocr-models.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path, name in assets:
                archive.write(path, name)
        write(asset_dir / 'evaluation-summary.json', summary)
        rows = ['# Aggregate OCR evaluation', '', summary['scope'], '', summary['protocol_note'], '',
                'Private images, labels, filenames and per-image predictions are not distributed.', '',
                '| Model | Color | Min width | Samples | Correct | Accuracy | CER |',
                '|---|---|---:|---:|---:|---:|---:|']
        for row in summary['evaluations']:
            rows.append(f"| {row['model']} | {row['color_order']} | {row['min_width']} | {row['samples']} | {row['correct']} | {row['exact_accuracy']:.2%} | {row['cer']:.2%} |")
        (source / 'docs/RESULTS.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
        (source / 'README.md').write_text(
            '# Vietnamese License Plate OCR — extended release\n\n'
            'This source profile accompanies separately distributed OCR weights and aggregate evaluation results. '
            'Publish it only after confirming code, OCR-weight and aggregate-result permissions, plus applicable third-party terms.\n\n'
            'See [aggregate results](docs/RESULTS.md), [setup](docs/SETUP.md), '
            '[model card](docs/MODEL_CARD.md) and [dataset card](docs/DATASET_CARD.md).\n\n'
            'Download `ocr-models.zip` from this repository’s approved GitHub Release, extract into the repository root, '
            'then install `requirements-ocr-cpu.txt`. Run `python scripts/generate_example.py` and '
            '`python -m plateocr infer data/examples/fictional_plate.png --model model92`.\n\n'
            'The OCR archive contains model92, model5plus and their dictionary. It does not contain YOLO weights, '
            'the original pretrained checkpoint, datasets, private labels or per-image predictions. '
            'Full vehicle-image detection requires a separately authorized detector.\n\n'
            'Architecture: BGR input [3, 48, 256] → MobileNetV3 → BiLSTM → CTC. '
            'The project provides dataset auditing, traceable train/eval/export commands, CLI, Streamlit and FastAPI.\n\n'
            'Run `python -m pip install -e ".[test]"` and `python -m unittest discover -s tests -v` for data-free tests. '
            'The synthetic example is not a quality benchmark.\n', encoding='utf-8')
        # Existing cards carry conservative context; add the profile-specific distribution clarification.
        for name in ('SETUP.md', 'MODEL_CARD.md', 'DATASET_CARD.md'):
            path = source / 'docs' / name
            text = path.read_text(encoding='utf-8')
            text = text.replace('Aggregate company-dataset metrics are withheld pending permission.',
                                'This extended profile provides reviewed aggregate metrics in RESULTS.md when permission is confirmed.')
            text = text.replace('Company-data metrics are withheld pending permission.',
                                'This extended profile includes aggregate metrics in RESULTS.md after permission confirmation.')
            text = text.replace('There is no public model URL while redistribution permission remains unknown.',
                                'For an authorized extended release, obtain the OCR archive from its GitHub Release attachments.')
            path.write_text('> Extended profile: OCR weights are separate release attachments; company data stays private.\n\n' + text, encoding='utf-8')
    manifest = {'schema_version': 1, 'profile': profile, 'permissions_confirmed': False,
                'files': {p.relative_to(bundle).as_posix(): digest(p) for p in sorted(bundle.rglob('*')) if p.is_file()}}
    write(bundle / 'bundle.json', manifest)
    write(root / '.private/releases' / ('latest-' + profile + '.json'), {'bundle': str(bundle.relative_to(root))})
    index = ['# Hai bản local — chưa công bố', '',
             'Mở README của bản muốn xem. Chỉ xuất bản sau khi xác nhận quyền tương ứng.', '']
    for mode in ('code-only', 'full'):
        pointer = root / '.private/releases' / ('latest-' + mode + '.json')
        if pointer.exists():
            folder = within(root, read(pointer)['bundle'])
            index.append(f'- **{mode}**: [README]({folder.name}/source/README.md)')
            if mode == 'full':
                index.append(f'  — [Kết quả tổng hợp]({folder.name}/source/docs/RESULTS.md), '
                             f'[Trọng số OCR]({folder.name}/assets/ocr-models.zip)')
    (root / '.private/releases/README.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return bundle


def verify(bundle):
    metadata = read(bundle / 'bundle.json')
    if metadata['profile'] not in ('code-only', 'full'):
        raise ValueError('Invalid profile')
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p != bundle / 'bundle.json'}
    if actual != set(metadata['files']):
        raise ValueError('Bundle file list changed; rebuild instead of editing it')
    for name, expected in metadata['files'].items():
        if digest(within(bundle, name)) != expected:
            raise ValueError('Bundle file changed: ' + name)
        rel = Path(name)
        if rel.parts[0] == 'source':
            if not allowed(Path(*rel.parts[1:]), metadata['profile']):
                raise ValueError('Unexpected source file: ' + name)
        elif metadata['profile'] != 'full' or name not in ('assets/ocr-models.zip', 'assets/evaluation-summary.json'):
            raise ValueError('Unexpected asset: ' + name)
    return metadata


def stage(profile, confirmed, root=ROOT):
    if not confirmed:
        raise ValueError('Staging for publication requires --confirm-permissions. Building previews does not.')
    pointer = read(root / '.private/releases' / ('latest-' + profile + '.json'))
    bundle = within(root, pointer['bundle'])
    metadata = verify(bundle)
    if metadata['profile'] != profile:
        raise ValueError('Profile mismatch')
    # Keep the user's main checkout and ignored local data untouched.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
    branch = 'publish/' + profile + '-' + stamp
    worktree = root / '.private/publish' / stamp
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '-b', branch, str(worktree), 'HEAD'], check=True)
    tracked = subprocess.check_output(['git', '-C', str(worktree), 'ls-files', '-z']).decode('utf-8').split('\0')
    source = bundle / 'source'
    selected = [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
    obsolete = [name for name in tracked if name and name not in selected]
    for name in obsolete:
        within(worktree, name)
    if obsolete:
        subprocess.run(['git', '-C', str(worktree), 'rm', '--', *obsolete], check=True)
    for name in selected:
        target = within(worktree, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    subprocess.run(['git', '-C', str(worktree), 'add', '--', *selected], check=True)
    diff = subprocess.run(['git', '-C', str(worktree), 'diff', '--cached', '--quiet'])
    if diff.returncode == 1:
        subprocess.run(['git', '-C', str(worktree), 'commit', '-m', 'release: prepare ' + profile + ' publication profile'], check=True)
    elif diff.returncode != 0:
        raise RuntimeError('Could not inspect staged changes')
    print('Local publication branch:', branch)
    print('Review checkout:', worktree)
    print('Nothing pushed. After review, run:')
    print(f'git -C "{worktree}" push -u origin "{branch}"')
    if profile == 'full':
        print('After merging this branch, attach these files to an authorized GitHub Release:', bundle / 'assets')
    return worktree


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('build', 'stage'):
        p = commands.add_parser(action)
        p.add_argument('profile', choices=('code-only', 'full'))
        if action == 'stage':
            p.add_argument('--confirm-permissions', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'build':
            print('Local preview created:', build(args.profile))
            print('Not published. Do not upload a full preview before permission is confirmed.')
        else:
            stage(args.profile, args.confirm_permissions)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
