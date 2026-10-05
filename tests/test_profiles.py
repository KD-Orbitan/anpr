import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import release_profiles as profiles


class ProfileTests(unittest.TestCase):
    def test_stage_uses_separate_branch_and_removes_obsolete_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init')
            git('config', 'user.name', 'Release test')
            git('config', 'user.email', 'release-test@example.invalid')
            (root / 'docs').mkdir()
            (root / 'README.md').write_text('original', encoding='utf-8')
            (root / 'docs/RESULTS.md').write_text('old results', encoding='utf-8')
            git('add', 'README.md', 'docs/RESULTS.md')
            git('commit', '-m', 'fixture')
            original_head = git('rev-parse', 'HEAD')
            bundle = root / '.private/releases/example'
            (bundle / 'source').mkdir(parents=True)
            (bundle / 'source/README.md').write_text('code-only', encoding='utf-8')
            profiles.write(bundle / 'bundle.json', {'profile': 'code-only', 'files': {
                'source/README.md': profiles.digest(bundle / 'source/README.md')}})
            profiles.write(root / '.private/releases/latest-code-only.json', {'bundle': '.private/releases/example'})
            worktree = profiles.stage('code-only', True, root)
            self.assertEqual(git('rev-parse', 'HEAD'), original_head)
            self.assertTrue((root / 'docs/RESULTS.md').exists())
            self.assertFalse((worktree / 'docs/RESULTS.md').exists())
            self.assertEqual((worktree / 'README.md').read_text(), 'code-only')

    def test_aggregate_projection_does_not_copy_private_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {'model': 'model92', 'samples': 10, 'correct': 9, 'failed_images': 0,
                      'exact_accuracy': 0.9, 'cer': 0.01, 'color_order': 'BGR', 'min_width': 1,
                      'limit': None, 'model_info': {'sha256': {'inference.pdiparams': 'testhash'}},
                      'manifest': 'CONFIDENTIAL_FILENAME', 'predictions': ['PRIVATE_PLATE'],
                      'company': 'PRIVATE_COMPANY'}
            profiles.write(root / 'report.json', report)
            profiles.write(root / 'inputs.json', {'metric_reports': ['report.json']})
            registry = {'models': {'model92': {'sha256': {'inference.pdiparams': 'testhash'}}}}
            result = profiles.summaries(root, root / 'inputs.json', registry)
            text = json.dumps(result)
            for marker in ('CONFIDENTIAL_FILENAME', 'PRIVATE_PLATE', 'PRIVATE_COMPANY', 'manifest'):
                self.assertNotIn(marker, text)
            self.assertEqual(result['evaluations'][0]['exact_accuracy'], 0.9)
            report['limit'] = 10
            profiles.write(root / 'report.json', report)
            with self.assertRaises(ValueError):
                profiles.summaries(root, root / 'inputs.json', registry)

    def test_staging_requires_explicit_confirmation(self):
        with self.assertRaises(ValueError):
            profiles.stage('full', False)

    def test_modified_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source').mkdir()
            path = root / 'source/README.md'
            path.write_text('source', encoding='utf-8')
            profiles.write(root / 'bundle.json', {'profile': 'code-only', 'files': {'source/README.md': profiles.digest(path)}})
            profiles.verify(root)
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                profiles.verify(root)

    def test_full_results_not_in_default_source_inventory(self):
        from check_release import allowed
        self.assertFalse(allowed(Path('docs/RESULTS.md')))
        self.assertTrue(allowed(Path('docs/RESULTS.md'), 'full'))

    def test_paths_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                profiles.within(Path(tmp), '../outside.txt')
