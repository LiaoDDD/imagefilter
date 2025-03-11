import torch
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import random  # 如果需要隨機抽取，可引入

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用裝置:", device)

model = YOLO("yolo11n.pt").to(device)

target_classes = {"bed", "dining table", "chair", "person"}

def load_urls(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

urls = load_urls("/app/test/testimg.txt")
print(f"讀取到 {len(urls)} 個圖片連結.")

# 若需隨機抽取 100 筆，取消下列註解：
# if len(urls) > 100:
#     urls = random.sample(urls, 100)
#     print("隨機抽取了 100 筆圖片連結。")

results_summary = [] 

for idx, url in enumerate(urls):
    print(f"\n處理圖片 {idx+1}/{len(urls)}: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"下載失敗：{response.status_code}")
            results_summary.append((url, None))
            continue
    except Exception as e:
        print(f"下載圖片時發生異常：{e}")
        results_summary.append((url, None))
        continue

    img_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        print("解碼失敗")
        results_summary.append((url, None))
        continue

    results = model(image)
    detected_targets = set()

    for result in results:
        for box in result.boxes:
            cls_idx = int(box.cls.cpu().numpy()[0])
            label = model.names[cls_idx]
            conf = float(box.conf.cpu().numpy()[0])
            print(f"檢測到: {label}, 置信度: {conf:.2f}")
            if label.lower() in target_classes and conf > 0.3:
                detected_targets.add(label.lower())
    
    if detected_targets:
        summary = f"圖片中有：{', '.join(detected_targets)}"
    else:
        summary = "圖片中沒有目標物件"
    
    print(summary)
    results_summary.append((url, detected_targets))
    

print("\n所有圖片的檢測結果：")
for url, detected in results_summary:
    if detected is None:
        print(f"{url}: 下載或解碼失敗")
    elif detected:
        print(f"{url}: {', '.join(detected)}")
    else:
        print(f"{url}: 無")
