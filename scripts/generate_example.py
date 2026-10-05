"""Generate a fictional OCR input; not a real plate or accuracy benchmark."""
from pathlib import Path
import cv2
import numpy as np

image = np.full((96, 420, 3), 245, dtype=np.uint8)
cv2.rectangle(image, (4, 4), (415, 91), (15, 15, 15), 2)
cv2.putText(image, '00A00000', (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (15, 15, 15), 3)
output = Path(__file__).resolve().parents[1] / 'data/examples/fictional_plate.png'
output.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(output), image):
    raise RuntimeError('Could not write example')
print(output)
