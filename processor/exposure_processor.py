# tools/exposure_processor.py
import cv2
import numpy as np
import logging

class ExposureProcessor:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def analyze(self, cv_image) -> dict:
        """
        接收OpenCV格式的圖片轉換為灰階後計算曝光與對比度
        返回對應的status
        """
        if cv_image is None:
            raise Exception("無效的圖片數據")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        high_light_ratio = np.sum(gray > 200) / gray.size * 100
        low_light_ratio = np.sum(gray < 50) / gray.size * 100

        if mean_brightness < 80 or low_light_ratio > 40:
            exposure_status = "underexposed"
        elif mean_brightness > 180 or high_light_ratio > 40:
            exposure_status = "overexposed"
        else:
            exposure_status = "properly"

        if std_brightness > 75:
            contrast_status = "high contrast"
        else:
            contrast_status = "properly"

        return {
            "exposure_status": exposure_status,
            "contrast_status": contrast_status
        }
