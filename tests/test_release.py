import importlib.util
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[1] / 'scripts/check_release.py'
spec = importlib.util.spec_from_file_location('check_release', script)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_confidential_and_generated_assets_are_not_exported(self):
        for name in ('data/active.json', 'data/versions/a/train.txt', 'data/private/image.jpg',
                     'runs/eval/predictions.csv', 'reports/model_evidence.json',
                     'docs/legacy_configs/mix5/config.yml', 'docs/AUDIT.md',
                     'models/model92/inference.pdiparams', 'configs/project.local.json',
                     '.private/backup.json', '.env', 'scripts/import_legacy.py'):
            with self.subTest(name=name):
                self.assertFalse(release.allowed(Path(name)))

    def test_public_sources_and_license_are_included(self):
        for name in ('README.md', 'plateocr/inference.py', 'models/characters.txt',
                     'configs/project.json', 'vendor/PaddleOCR/LICENSE', 'docs/MODEL_CARD.md'):
            self.assertTrue(release.allowed(Path(name)))

    def test_relative_manifest_survives_relocation(self):
        from plateocr.data import read_manifest
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                root = Path(directory)
                (root / 'labels.txt').write_text('images/synthetic.jpg\t00A00000\n', encoding='utf-8')
                record = read_manifest(root / 'labels.txt', root)[0]
            # Windows runners may expose TEMP through its DOS 8.3 alias.
            self.assertEqual(Path(record['image']).parent, (root / 'images').resolve())
