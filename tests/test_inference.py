import unittest

try:
    import numpy as np
    import cv2
except ImportError:
    np = None

from plateocr.inference import decode_ctc, preprocess


@unittest.skipIf(np is None, "Requires numpy and OpenCV")
class InferenceTests(unittest.TestCase):
    def test_ctc_preserves_repeats_separated_by_blank(self):
        indices = [1, 1, 0, 1, 2, 2, 0]
        probabilities = np.eye(3, dtype="float32")[indices][None, ...]
        self.assertEqual(decode_ctc(probabilities, ["A", "B"]), [("AAB", 1.0)])

    def test_ctc_blank_and_dictionary_mismatch(self):
        self.assertEqual(decode_ctc(np.array([[[1, 0, 0]]]), ["A", "B"]), [("", 0.0)])
        with self.assertRaises(ValueError):
            decode_ctc(np.zeros((1, 10, 5)), ["A"])

    def test_preprocessing_preserves_bgr_and_padding(self):
        image = np.zeros((48, 48, 3), dtype="uint8")
        image[:, :, 0] = 255
        tensor = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 48, 256))
        self.assertTrue(np.all(tensor[0, 0, :, :48] == 1))
        self.assertTrue(np.all(tensor[0, 2, :, :48] == -1))
        self.assertTrue(np.all(tensor[:, :, :, 48:] == 0))


if __name__ == "__main__":
    unittest.main()
