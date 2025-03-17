import imagehash
import requests
from io import BytesIO
from PIL import Image

def compute_phash(url: str, timeout: int = 5):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return imagehash.phash(img)
    except Exception as e:
        print(f"Error computing pHash for {url}: {e}")
        return None

if __name__ == "__main__":
 
    url1 = ""
    url2 = ""
    
    print("下載並計算圖片")
    phash1 = compute_phash(url1)
    phash2 = compute_phash(url2)
    
    if phash1 is None or phash2 is None:
        print("計算失敗")
    else:
        print(f"Image 1 pHash: {phash1}")
        print(f"Image 2 pHash: {phash2}")

        hamming_distance = phash1 - phash2
        print(f"Hamming Distance: {hamming_distance}")
 
        if hamming_distance <= 5:
            print("相似度高")
        else:
            print("相似度低。")
