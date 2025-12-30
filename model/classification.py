import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pickle
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_TRANSLATION_PATH = os.path.join(_DATA_DIR, "breed_translation.json")

with open(_TRANSLATION_PATH, encoding="utf-8") as f:
    BREED_TRANSLATION = json.load(f)

def crop_image_by_bbox(image_path: str, bbox):
    """
    Обрезает изображение по bounding box и возвращает PIL Image.
    
    Args:
        image_path (str): Путь к исходному изображению
        bbox (tuple): (x1, y1, x2, y2) — координаты рамки
    
    Returns:
        PIL.Image.Image: Обрезанное изображение
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB") 
        return img.crop(bbox)

class BreedClassifier:
    """
    Классификатор пород собак на основе предобученной ResNet18 + Logistic Regression.
    Использует torchvision.models.resnet18(weights='IMAGENET1K_V1') напрямую.
    Требуется только файл breed_classifier_best.pkl.
    """
    
    def __init__(self):
        """
        Инициализирует модель.
        
        Args:
            model_dir (str): Путь к папке с файлом:
                - breed_classifier_best.pkl
        """
        print(f"DEBUG: __file__ = {__file__}")
        print(f"DEBUG: os.path.dirname(__file__) = {os.path.dirname(__file__)}")
        model_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
        print(f"DEBUG: model_dir = {os.path.abspath(model_dir)}")
        clf_path = os.path.join(model_dir, "breed_classifier_best.pkl")
        print(f"DEBUG: clf_path = {os.path.abspath(clf_path)}")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Не найден файл классификатора: {clf_path}")

        model_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
        self.device = torch.device("cpu")
        
        # --- Загружаем предобученную ResNet18 напрямую из torchvision ---
        self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = torch.nn.Identity()  # удаляем классификатор ImageNet
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # --- Загрузка только классификатора (LogisticRegression) ---
        clf_path = os.path.join(model_dir, "breed_classifier_best.pkl")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Не найден файл классификатора: {clf_path}")
        
        with open(clf_path, "rb") as f:
            data = pickle.load(f)
            self.classifier = data["classifier"]
            self.class_names = data["class_names"]
        
        # --- Трансформации (должны совпадать с обучением!) ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],    # Средние значения ImageNet
                std=[0.229, 0.224, 0.225]      # Стандартные отклонения ImageNet
            )
        ])
    
    def predict(self, pil_image):
        """
        Предсказывает породу собаки на изображении.
        
        Args:
            pil_image (PIL.Image.Image): Изображение в формате PIL (RGB).
            
        Returns:
            tuple: (порода: str, уверенность: float)
        """
        # Предобработка изображения
        img_t = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Извлечение признаков через предобученную ResNet18
        with torch.no_grad():
            features = self.backbone(img_t).cpu().numpy()
        
        # Классификация
        probabilities = self.classifier.predict_proba(features)[0]
        predicted_class_id = self.classifier.predict(features)[0]
        
        # Преобразуем ID в читаемое имя породы
        raw_breed_name = self.class_names[predicted_class_id]

        # Преобразуем ID в читаемое имя породы (ключ для словаря)
        if "-" in raw_breed_name:
            breed_key = raw_breed_name.split("-", 1)[1].lower()  # например: "chihuahua"
        else:
            breed_key = raw_breed_name.replace("_", " ").lower()  # на всякий случай

        # Получаем перевод (или оставляем ключ, если перевода нет)
        breed_name = BREED_TRANSLATION.get(breed_key, breed_key.replace("_", " ").title())
        
        confidence = float(probabilities[predicted_class_id])
        return breed_name, confidence