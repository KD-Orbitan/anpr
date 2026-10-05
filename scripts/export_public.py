"""Create an exclusive, allowlisted source release for review before publishing."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from check_release import ROOT, check, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    issues = check()
    if issues:
        raise SystemExit('\n'.join(issues))
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise SystemExit('Choose a new output folder outside the project; nothing will be overwritten.')
    output.mkdir(parents=True, exist_ok=False)
    entries = {}
    for source in files():
        relative = source.relative_to(ROOT)
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        entries[relative.as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f'Exported {len(entries)} allowlisted source files to {output}')
    # Inventory is stored outside the release tree so it cannot accidentally expose local files.
    inventory = ROOT / '.private' / (output.name + '-inventory.json')
    inventory.parent.mkdir(exist_ok=True)
    inventory.write_text(json.dumps(entries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
