"""Verify authorized local weight files without downloading any artifacts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plateocr.project import ROOT, read_json, resolve, settings, sha256


def main():
    cfg = settings()
    registry = read_json(ROOT / 'models/registry.json')
    assets = [(resolve(cfg['dictionary']), registry['dictionary_sha256']),
              (resolve(cfg['detector']), registry['detector']['sha256'])]
    for entry in registry['models'].values():
        assets.extend((resolve(entry['path']) / name, digest) for name, digest in entry['sha256'].items()
                      if name in ('inference.pdmodel', 'inference.pdiparams'))
    failed = False
    for path, expected in assets:
        status = 'missing' if not path.is_file() else ('ok' if sha256(path) == expected else 'hash mismatch')
        print(status, path.relative_to(ROOT))
        failed |= status != 'ok'
    if failed:
        print('See docs/SETUP.md. Weights are not bundled and no public download has been approved.')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
