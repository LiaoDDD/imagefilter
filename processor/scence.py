import requests
import torch
from PIL import Image
import torch.nn as nn
from io import BytesIO
from torchvision import transforms
from torchvision.models import resnet18

def load_365_model(model_weight, label_file):
    model = resnet18(num_classes=365)

    checkpoint = torch.load(model_weight, map_location='cuda')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
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
            category = parts[0]
            classes.append(category)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, classes, tfm

def analyze_scene(image_url, model, tfm, classes):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert('RGB')
    img_tensor = tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)

    top5_prob, top5_id = torch.topk(probs, 5)
    top5_prob = top5_prob.squeeze().tolist()
    top5_id = top5_id.squeeze().tolist()

    results = []
    for i in range(5):
        class_id = top5_id[i]
        results.append((classes[class_id], top5_prob[i] * 100))
    return results

def load_url(file_path):
    with open(file_path) as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

if __name__ == '__main__':
    model_weight = '/app/test/resnet18_places365.pth.tar'
    label_file = '/app/test/categories_places365.txt'
    url_path = '/app/test/testimg.txt'

    # 設定門檻值 threshold
    threshold = 40.0  

    model, classes, tfm = load_365_model(model_weight, label_file)

    urls = load_url(url_path)
    print(f"讀取到 {len(urls)} 個圖片")

    results_summary = []
    count_top1_above_threshold = 0

    for idx, url in enumerate(urls):
        print(f"\n處理圖片 {idx+1}/{len(urls)}: {url}")
        try:
            top5_results = analyze_scene(url, model, tfm, classes)
            print("分類結果：")
            for rank, (scene, prob) in enumerate(top5_results, start=1):
                print(f"Rank {rank}: {scene} | 機率 = {prob:.2f}%")

            # 檢查 Top1 預測機率是否超過 threshold
            if top5_results and top5_results[0][1] > threshold:
                count_top1_above_threshold += 1
                print(
                    f"*** 這張圖的 Top1 場景為「{top5_results[0][0]}」"
                    f"，機率約 {top5_results[0][1]:.2f}%，已超過 {threshold}% ***"
                )

            results_summary.append((url, top5_results))

        except Exception as e:
            print(f"處裡失敗: {e}")
            results_summary.append((url, None))

    print("\n=== 最終檢測結果彙整 ===")
    for url, result in results_summary:
        if result is None:
            print(f"{url}：處理失敗")
        else:
            # 取得 Top1
            top1_scene, top1_prob = result[0]

            # 如果 Top1 機率低於 threshold，就把場景視作 None
            if top1_prob < threshold:
                print(f"{url}: None (Top1 機率僅 {top1_prob:.2f}%)")
            else:
                # 否則照常印出場景
                print(f"{url} : Top1 場景為「{top1_scene}」，機率約 {top1_prob:.2f}%")
            print(f"→ Top5 結果: {result}")

    print(
        f"\nTop1 預測機率超過 {threshold}% 的圖片數量："
        f"{count_top1_above_threshold} / {len(urls)}"
    )
