import torch
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用裝置:", device)

model = YOLO("yolo11n.pt").to(device)

target_classes = {
    "bed", "dining table", "chair", "person", "bottle", "cup",
    "couch", "toilet", "tv", "oven", "hair drier"
}

# 確保輸出資料夾存在
output_dir = "/app/test/yolo_output"
os.makedirs(output_dir, exist_ok=True)

def load_urls(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

urls = load_urls("test/urls_data.txt")
print(f"讀取到 {len(urls)} 個圖片連結.")

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
            xyxy = box.xyxy.cpu().numpy()[0].astype(int)
            x1, y1, x2, y2 = xyxy

            print(f"檢測到: {label}, 置信度: {conf:.2f}")

            if label.lower() in target_classes and conf > 0.45:
                detected_targets.add(label.lower())

                # 畫框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 加上類別與置信度標籤
                label_text = f"{label} {conf:.2f}"
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.8, (0, 255, 0), 2)

    # 儲存標註後的圖片
    filename = f"image_{idx+1}.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"標註圖片已儲存至：{save_path}")

    if detected_targets:
        summary = f"圖片中有：{', '.join(detected_targets)}"
    else:
        summary = "圖片中沒有目標物件"

    print(summary)
    results_summary.append((url, detected_targets))

# 檢測結果彙總
print(f"\n所有圖片的檢測結果彙總:")
for url, detected in results_summary:
    if detected is None:
        print(f"{url}: 下載或解碼失敗")
    elif detected:
        print(f"{url}: {', '.join(detected)}")
    else:
        print(f"{url}: 無")
