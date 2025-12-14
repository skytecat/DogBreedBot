# model/detection.py
import cv2
import shutil
from ultralytics import YOLO

# Загружаем модель один раз
_model = YOLO("yolov8n.pt")

def detect_best_dog_bbox(image_path: str):
    """
    Возвращает:
      - (bbox, confidence) — координаты и уверенность самой уверенной собаки,
      - или (None, None), если собак нет.
    """
    results = _model(image_path, verbose=False)
    boxes = results[0].boxes

    # Находим ID класса "dog"
    dog_class_id = None
    for cls_id, name in _model.names.items():
        if name == "dog":
            dog_class_id = cls_id
            break

    if dog_class_id is None:
        return None, None

    # Фильтруем только собак
    dog_boxes = [box for box in boxes if int(box.cls.item()) == dog_class_id]
    if not dog_boxes:
        return None, None

    # Выбираем самую уверенную
    best_box = max(dog_boxes, key=lambda b: b.conf.item())
    bbox = tuple(map(int, best_box.xyxy[0].tolist()))
    confidence = best_box.conf.item()
    return bbox, confidence


def draw_dog_bbox(image_path: str, output_path: str):
    """
    Рисует bbox с надписью "Dog {confidence:.0%}".
    Возвращает: (found: bool, confidence: float or None)
    """
    bbox, confidence = detect_best_dog_bbox(image_path)
    
    if bbox is None:
        shutil.copy(image_path, output_path)
        return False, None

    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Форматируем уверенность: например, "Dog 92%"
    label = f"Dog {confidence:.0%}"
    cv2.putText(
        img, label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 0, 255), 2
    )
    cv2.imwrite(output_path, img)
    return True, confidence