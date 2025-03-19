import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from io import BytesIO
import requests
import logging

class SceneProcessor:
    def __init__(self, model_weight: str, label_file: str, threshold: float = 40.0, device: str = None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.model, self.classes, self.tfm = self.load_model(model_weight, label_file, device)

    def load_model(self, model_weight: str, label_file: str, device: str):
        model = resnet18(num_classes=365)
        checkpoint = torch.load(model_weight, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        classes = []
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # 這裡取第一個字段作為分類名稱
                category = parts[0]
                classes.append(category)
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return model, classes, tfm

    def analyze(self, image_url: str) -> dict:
        """
        根據圖片 URL 下載圖片，並利用 Places365 模型分析場景。
        若 Top1 預測機率大於 threshold返回 {"scene": <場景>, "probability": <機率>}
        否則返回 None。
        """
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_tensor = self.tfm(img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
        top1_prob, top1_id = torch.topk(probs, 1)
        top1_prob = top1_prob.squeeze().item() * 100  # 轉為百分比
        top1_id = top1_id.squeeze().item()
        if top1_prob >= self.threshold:
            return {"scene": self.classes[top1_id], "probability": top1_prob}
        else:
            return None

