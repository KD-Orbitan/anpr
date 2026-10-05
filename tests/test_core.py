import tempfile
import unittest
from pathlib import Path

from plateocr.data import read_manifest
from plateocr.evaluation import edit_distance, metrics
from plateocr.dataset import group_split


class EvaluationTests(unittest.TestCase):
    def test_failed_images_are_not_dropped_from_accuracy(self):
        result = metrics([
            {"label": "30A12345", "prediction": "30A12345", "error": ""},
            {"label": "30A12345", "prediction": "", "error": "missing"},
        ])
        self.assertEqual(result["exact_accuracy"], 0.5)
        self.assertEqual(result["cer"], 0.5)
        self.assertEqual(result["failed_images"], 1)

    def test_edit_distance_insert_delete_substitute(self):
        self.assertEqual(edit_distance("AB12", "AB123"), 1)
        self.assertEqual(edit_distance("AB123", "AB12"), 1)
        self.assertEqual(edit_distance("AB123", "AC123"), 1)

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            metrics([])


class ManifestTests(unittest.TestCase):
    def test_dictionary_bytes_match_registry(self):
        from plateocr.project import ROOT, read_json, sha256
        registry = read_json(ROOT / 'models/registry.json')
        self.assertEqual(sha256(ROOT / 'models/characters.txt'), registry['dictionary_sha256'])

    def test_split_groups_plate_variants_and_is_repeatable(self):
        records = [{"image": f"{i}-{j}.jpg", "label": str(i)} for i in range(40) for j in range(3)]
        splits = group_split(records)
        self.assertEqual(splits, group_split(records))
        labels = {name: {r["label"] for r in rows} for name, rows in splits.items()}
        self.assertFalse(labels["train"] & labels["val"])
        self.assertFalse(labels["train"] & labels["test"])
        self.assertFalse(labels["val"] & labels["test"])
        self.assertEqual(sum(map(len, splits.values())), len(records))

    def test_duplicate_paths_and_malformed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            for content in ("a.jpg\tABC\na.jpg\tDEF\n", "a.jpg ABC\n", "a.jpg\t\n"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_manifest(path, tmp)

    def test_relative_path_is_resolved_against_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            path.write_text("images/a.jpg\t30A12345\n", encoding="utf-8")
            result = read_manifest(path, tmp)
            self.assertEqual(result[0]["image"], str((Path(tmp) / "images/a.jpg").resolve()))


if __name__ == "__main__":
    unittest.main()
