import cv2
from ultralytics import YOLO

# Загружаем модель
_model = YOLO("yolov8n.pt")

def detect_best_dog_bbox(image_path: str):
    """
    Возвращает:
      - (x1, y1, x2, y2) — координаты bbox собаки c самой большой уверенностью, 
      - или None, если собак не найдено
    """
    results = _model(image_path, verbose=False)
    boxes = results[0].boxes
    
    # Ищем ID класса "dog"
    dog_class_id = None
    for cls_id, name in _model.names.items():
        if name == "dog":
            dog_class_id = cls_id
            break
    
    if dog_class_id is None:
        return None
    
    # Фильтруем собак и выбираем с самой большой уверенностью
    dog_boxes = [
        box for box in boxes 
        if int(box.cls.item()) == dog_class_id
    ]
    
    if not dog_boxes:
        return None
    
    best_box = max(dog_boxes, key=lambda b: b.conf.item())
    return tuple(map(int, best_box.xyxy[0].tolist()))


def draw_dog_bbox(image_path: str, output_path: str):
    """
    Рисует bbox собаки с самой большой уверенностью и сохраняет результат.
    Возвращает: True, если собака найдена, иначе False
    """
    bbox = detect_best_dog_bbox(image_path)
    if bbox is None:
        # Просто копируем исходное изображение
        import shutil
        shutil.copy(image_path, output_path)
        return False
    
    # Загружаем и рисуем
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        img, "Dog", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
    )
    cv2.imwrite(output_path, img)
    return True