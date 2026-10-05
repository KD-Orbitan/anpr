import math
from pathlib import Path

from .project import ROOT, model_info, read_json, resolve, settings, sha256


def decode_ctc(output, characters):
    """Greedy CTC; collapse adjacent repeats BEFORE removing blanks."""
    if output.ndim != 3 or output.shape[2] != len(characters) + 1:
        raise ValueError("OCR output shape does not match the character dictionary")
    results = []
    for sample in output:
        indices = sample.argmax(axis=1)
        probs = sample.max(axis=1)
        text, scores, previous = [], [], None
        for index, score in zip(indices, probs):
            index = int(index)
            if index != 0 and index != previous:
                text.append(characters[index - 1])
                scores.append(float(score))
            previous = index
        results.append(("".join(text), sum(scores) / len(scores) if scores else 0.0))
    return results


def preprocess(image, color_order="BGR", min_width=1):
    import cv2
    import numpy as np
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a non-empty three-channel BGR image")
    if color_order == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_order != "BGR":
        raise ValueError("color_order must be BGR or RGB")
    h, w = image.shape[:2]
    width = min(256, max(min_width, math.ceil(48 * w / h)))
    resized = cv2.resize(image, (width, 48), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype("float32").transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    padded = np.zeros((3, 48, 256), dtype="float32")
    padded[:, :, :width] = normalized
    return padded[None, ...]


class Recognizer:
    def __init__(self, model=None, color_order="BGR", legacy_preprocess=False):
        from paddle.inference import Config, create_predictor
        cfg = settings()
        if cfg["image_shape"] != [3, 48, 256]:
            raise ValueError("This project supports only CRNN input [3, 48, 256]")
        self.name = model or cfg["default_model"]
        self.info = model_info(self.name)
        registry = read_json(ROOT / "models/registry.json")
        if sha256(resolve(cfg["dictionary"])) != registry["dictionary_sha256"]:
            raise ValueError("Character dictionary differs from registered models")
        self.characters = resolve(cfg["dictionary"]).read_text(encoding="utf-8-sig").splitlines()
        self.color_order = color_order
        self.min_width = 32 if legacy_preprocess else 1
        folder = resolve(self.info["path"])
        for name in ("inference.pdmodel", "inference.pdiparams"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
            if sha256(folder / name) != self.info["sha256"][name]:
                raise ValueError("Model asset differs from registry: " + str(folder / name))
        config = Config(str(folder / "inference.pdmodel"), str(folder / "inference.pdiparams"))
        config.disable_gpu()
        config.disable_glog_info()
        config.enable_memory_optim()
        config.set_cpu_math_library_num_threads(2)
        self.predictor = create_predictor(config)

    def predict(self, image):
        value = preprocess(image, self.color_order, self.min_width)
        self.predictor.get_input_handle(self.predictor.get_input_names()[0]).copy_from_cpu(value)
        self.predictor.run()
        output = self.predictor.get_output_handle(self.predictor.get_output_names()[0]).copy_to_cpu()
        text, confidence = decode_ctc(output, self.characters)[0]
        return {"text": text, "confidence": confidence}


def read_image(path):
    import cv2
    import numpy as np
    # imdecode handles Unicode filenames on Windows.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image: " + str(path))
    return image


class Pipeline:
    def __init__(self, model=None, detector_conf=0.25):
        from ultralytics import YOLO
        self.detector = YOLO(str(resolve(settings()["detector"])))
        self.ocr = Recognizer(model)
        self.detector_conf = detector_conf

    def predict(self, image):
        result = self.detector(image, conf=self.detector_conf, verbose=False)[0]
        plates = []
        h, w = image.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            plates.append({"box": [x1, y1, x2, y2], "detection_confidence": float(box.conf[0]),
                           **self.ocr.predict(image[y1:y2, x1:x2])})
        return sorted(plates, key=lambda item: item["detection_confidence"], reverse=True)
