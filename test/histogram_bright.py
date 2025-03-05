import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def histogram_analysis(img_url):
    response = requests.get(img_url)
    if response.status_code != 200:
        raise Exception(f"下載失敗: {response.status_code}")
    
    img_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception("fail")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.size

    high_pixels = np.sum(hist[200:]) / total_pixels * 100
    return hist, high_pixels
hist, high_percentage = histogram_analysis("https://i.travelapi.com/lodging/8000000/7100000/7098000/7097957/7c879bf1.jpg")
print("高亮像素比例 (%):", high_percentage)
# 可選：顯示直方圖
plt.plot(hist)
plt.title("gray histogram")
plt.xlabel("bright")
plt.ylabel("pixels")
plt.show()
