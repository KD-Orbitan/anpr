import base64
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from api import app
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Requires application environment")
class ApiTests(unittest.TestCase):
    def test_health_and_invalid_requests(self):
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        for value in ("", "invalid", base64.b64encode(b"not an image").decode()):
            self.assertEqual(client.post("/anpr", json={"image_base64": value}).status_code, 400)
        self.assertEqual(client.post("/anpr", json={}).status_code, 422)

    def test_decoded_image_is_sent_to_pipeline(self):
        client = TestClient(app)
        image = np.zeros((12, 20, 3), dtype="uint8")
        image[:, :, 0] = 255
        _, encoded = cv2.imencode(".png", image)
        with patch("api.pipeline") as factory:
            factory.return_value.predict.return_value = [{"text": "30A12345", "confidence": 0.9}]
            response = client.post("/anpr", json={"image_base64": base64.b64encode(encoded).decode()})
            self.assertEqual(response.status_code, 200)
            np.testing.assert_array_equal(factory.return_value.predict.call_args.args[0], image)
            self.assertEqual(response.json()["plates"][0]["text"], "30A12345")

    def test_crop_endpoint_does_not_load_detector(self):
        client = TestClient(app)
        image = np.full((48, 256, 3), 125, dtype="uint8")
        _, encoded = cv2.imencode(".png", image)
        with patch("api.recognizer") as factory, patch("api.pipeline") as detector:
            factory.return_value.predict.return_value = {"text": "00A00000", "confidence": 0.8}
            response = client.post("/ocr", json={"image_base64": base64.b64encode(encoded).decode()})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["plate"]["text"], "00A00000")
            np.testing.assert_array_equal(factory.return_value.predict.call_args.args[0], image)
            detector.assert_not_called()
        self.assertEqual(client.post("/ocr", json={"image_base64": "invalid"}).status_code, 400)
