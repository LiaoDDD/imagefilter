import cv2
import numpy as np
import requests

def analyze_exposure_from_url(image_url):

    response = requests.get(image_url, timeout=10)
    if response.status_code != 200:
        raise Exception("下載失敗：" + str(response.status_code))
    
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception("失敗")
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
   
    mean_brightness = np.mean(gray)        
    std_brightness = np.std(gray)           
    high_light_ratio = np.sum(gray > 200) / gray.size * 100  # 高亮像素比例
    low_light_ratio = np.sum(gray < 50) / gray.size * 100    # 低亮像素比例
    exposure_index = mean_brightness / (std_brightness + 1e-5) 
    
    if mean_brightness < 80 or low_light_ratio > 40:
        exposure_status = "過暗"
    elif mean_brightness > 180 or high_light_ratio > 40:
        exposure_status = "過亮"
    else:
        exposure_status = "正常"
    
   
    if std_brightness > 75:
        contrast_status = "對比度過高"
    else:
        contrast_status = "正常對比度"
    

    return {
        "曝光狀態": exposure_status,
        "對比度狀況": contrast_status
    }

def load_urls(file_path):
    
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

if __name__ == '__main__':
  
    file_path = "/app/test/testimg.txt"
    urls = load_urls(file_path)
    print(f"圖取到 {len(urls)} 圖片。")
    
    results_summary = []
    
    for idx, url in enumerate(urls):
        print(f"\n處理圖片 {idx+1}/{len(urls)}: {url}")
        try:
            result = analyze_exposure_from_url(url)
            print(f"分析結果: {result}")
            results_summary.append((url, result))
        except Exception as e:
            print(f"處裡失敗: {e}")
            results_summary.append((url, None))
    
    print("\n最終檢測結果：")
    for url, result in results_summary:
        if result is None:
            print(f"{url}：處理失敗")
        else:
            print(f"{url}：{result}")
