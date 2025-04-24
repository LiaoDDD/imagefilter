import asyncio
import aiohttp
import os
from io import BytesIO
from PIL import Image
import logging
import imagehash
from dotenv import load_dotenv
load_dotenv()

class ImageProcessor:
    def __init__(self):
        # 從環境變數讀取並發數量（預設 100）
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", 100))
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def fetch_image(self, session, url, retries=3):
        for attempt in range(retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logging.warning(f"下載失敗：{url} (狀態碼: {response.status})")
                        return None
            except aiohttp.ClientPayloadError as e:
                logging.error(f"下載圖片例外：{url}, 嘗試 {attempt+1}/{retries} 次失敗，錯誤：{e}")
                if attempt == retries - 1:
                    return None
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"下載圖片例外：{url}, 錯誤：{e}")
                return None

    async def process_image(self, session, url):
        ext = os.path.splitext(url)[1].lower()
        image_data = await self.fetch_image(session, url)
        if image_data is None:
            return {
                "url": url,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "dpi": None,
                "extension": ext,
                "phash": None,
                "raw": None
            }
        try:
            with Image.open(BytesIO(image_data)) as img:
                width, height = img.size
                # 計算感知哈希，作為重複檢測依據
                phash = str(imagehash.phash(img))
                if height != 0:
                    if width < height:
                        aspect_ratio = height / width
                    else:
                        aspect_ratio = width / height
                else:
                    aspect_ratio = None
                dpi = img.info.get("dpi", (None, None))[0]
                return {
                    "url": url,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "dpi": dpi,
                    "extension": ext,
                    "phash": phash,
                    "raw": image_data
                }
        except Exception as e:
            logging.error(f"處理圖片失敗：{url}, 錯誤：{e}")
            return {
                "url": url,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "dpi": None,
                "extension": ext,
                "phash": None,
                "raw": None
            }

    async def process_images(self, urls: list):
        results = []
        async with aiohttp.ClientSession() as session:
            async def sem_task(url):
                async with self.semaphore:
                    return await self.process_image(session, url)
            tasks = [sem_task(url) for url in urls]
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
        return results

