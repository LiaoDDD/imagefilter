import os
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models

def load_trained_model(model_path, device, num_classes):
    # 以 ResNet18 為例，根據需要可自行修改
    model = models.resnet18(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def predict_image_from_url(model, image_url, transform, device, threshold=0.8, show_image=True):
    # 下載圖片
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except Exception as e:
        print(f"下載圖片失敗: {e}")
        return None

    # 解析圖片並轉換為 RGB
    try:
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"解析圖片失敗: {e}")
        return None

    # 預處理並轉換成 Tensor
    input_tensor = transform(pil_img)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        target_probs = probs[:, [1, 2]]
        max_prob, target_idx = torch.max(target_probs, dim=1)
    conf_value = max_prob.item()

    if conf_value < threshold:
        result_str = f"Uncertain (max prob: {conf_value:.2f})"
    else:
        if target_idx.item() == 0:
            result_str = f"Natural Light (max prob: {conf_value:.2f})"
        else:
            result_str = f"Artificial Light (max prob: {conf_value:.2f})"

    if show_image:
        img_np = np.array(pil_img)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.axis('off')
        plt.text(10, 30, result_str, color='red', fontsize=14,
                 bbox=dict(facecolor='black', alpha=0.7))
        plt.tight_layout()
        plt.show()

    return result_str

def predict_from_url_file(model, txt_filepath, transform, device, threshold=0.8, show_image=True):
    results = {}
    try:
        with open(txt_filepath, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"讀取 URL 檔案失敗: {e}")

    for url in urls:
        result = predict_image_from_url(model, url, transform, device, threshold=threshold, show_image=show_image)
        results[url] = result
    return results

def main():
    # 透過環境變數讀取參數
    model_path = os.environ.get("MODEL_PATH", "/app/test/best_udcsit_model.pth")
    url_file = os.environ.get("URL_FILE", "/app/test/urls_data.txt")
    num_classes = int(os.environ.get("NUM_CLASSES", "4"))
    threshold = float(os.environ.get("THRESHOLD", "0.8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 與訓練時相同的圖片預處理流程
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 載入模型
    model = load_trained_model(model_path, device, num_classes)

    # 讀取 URL 檔案
    try:
        with open(url_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"讀取 URL 檔案失敗: {e}")
        return

    print(f"讀取到 {len(urls)} 個圖片 URL")

    # 依序處理每個圖片 URL
    predictions = {}
    above_count = 0
    for url in urls:
        print(f"\n處理圖片: {url}")
        result = predict_image_from_url(model, url, data_transforms, device, threshold=threshold, show_image=True)
        predictions[url] = result
        if result and not result.startswith("Uncertain"):
            above_count += 1
        print(f"預測結果: {result}")

    print("\n=== 預測結果彙整 ===")
    for url, pred in predictions.items():
        print(f"{url} --> {pred}")

    print(f"{above_count} / {len(urls)}")

if __name__ == "__main__":
    main()
