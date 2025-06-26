from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

def detect_license_plates(image):
    results = model.predict(source=image, save=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image[y1:y2, x1:x2]
            boxes.append(((x1, y1, x2, y2), cropped))
    return boxes
