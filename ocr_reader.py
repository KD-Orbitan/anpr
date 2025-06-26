from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='vi')

def read_text_from_image(image):
    text_lines = []
    cropped_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_result = ocr.predict(cropped_rgb)[0]
    rec_texts = ocr_result.get('rec_texts', [])
    rec_scores = ocr_result.get('rec_scores', [])

    for text, score in zip(rec_texts, rec_scores):
        text_lines.append(text)
    full_text = ' '.join(text_lines) if text_lines else "No text detected"
    return full_text
