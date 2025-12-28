import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pickle
import os

class BreedClassifier:
    """
    Классификатор пород собак на основе предобученной ResNet18 + Logistic Regression.
    Использует torchvision.models.resnet18(weights='IMAGENET1K_V1') напрямую.
    Требуется только файл breed_classifier_best.pkl.
    """
    
    def __init__(self, model_dir="../weights"):
        """
        Инициализирует модель.
        
        Args:
            model_dir (str): Путь к папке с файлом:
                - breed_classifier_best.pkl
        """
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
        if "-" in raw_breed_name:
            breed_name = raw_breed_name.split("-", 1)[1].replace("_", " ").title()
        else:
            breed_name = raw_breed_name.replace("_", " ").title()
        
        confidence = float(probabilities[predicted_class_id])
        return breed_name, confidence