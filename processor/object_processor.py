import torch
from ultralytics import YOLO
import numpy as np
import logging

class ObjectProcessor:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo11n.pt").to(device)
        # 設定只關注目標物件（例如 bed 與 chair），根據需求擴充
        self.target_objects = {"bed", "chair", "person"}

    def process(self, image: np.ndarray) -> list:
        """
        傳入 OpenCV 格式的圖片（BGR），執行物件偵測，
        返回偵測到的目標物件列表（小寫）。
        """
        try:
            results = self.model(image)
            detected = set()
            for result in results:
                for box in result.boxes:
                    cls_idx = int(box.cls.cpu().numpy()[0])
                    label = self.model.names[cls_idx]
                    conf = float(box.conf.cpu().numpy()[0])
                    if label.lower() in self.target_objects and conf > 0.3:
                        detected.add(label.lower())
            return list(detected)
        except Exception as e:
            logging.error(f"物件偵測失敗: {e}")
            return []
