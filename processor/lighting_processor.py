import torch
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import cv2
from io import BytesIO
import requests
import logging

class LightingProcessor:
    def __init__(self, model_path: str, label_file: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.load_model(model_path, device)
        self.labels = self.load_labels(label_file)
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str, device: str):
        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict):
            # 假設模型架構為 ResNet18，你可以根據實際情況修改
            model = models.resnet18(num_classes=4)
            model.load_state_dict(loaded_obj)
        else:
            model = loaded_obj
        model = model.to(device)
        model.eval()
        return model

    def load_labels(self, label_file: str):
        with open(label_file, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels

    def convert_cv2_to_pil(self, cv_image):
        try:
            # 轉換 BGR 到 RGB
            cv2_im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_im_rgb)
            return pil_img
        except Exception as e:
            logging.error(f"圖片格式轉換失敗: {e}")
            raise e

    def predict_from_cv2(self, cv_image, threshold=0.8) -> str:
        """
        接收 OpenCV 格式圖片，內部完成轉換並進行預測返回光線標籤。
        """
        pil_img = self.convert_cv2_to_pil(cv_image)
        return self.predict(pil_img)

    def predict(self, img: Image.Image, threshold=0.8) -> str:
        x = self.tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            # 僅取索引 1 與 2 的機率
            target_probs = probs[:, [1, 2]]
            max_prob, target_idx = torch.max(target_probs, dim=1)
        conf_value = max_prob.item()

        if conf_value < threshold:
            return ""  # 置信度太低，不返回標籤
        else:
            if target_idx.item() == 0:
                return f"natural_light"
            else:
                return f"indoor_light"

    def predict_from_url(self, image_url: str,threshold=0.8) -> str:
        # 略過下載邏輯，直接轉換後預測
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return self.predict(img, threshold=threshold)
