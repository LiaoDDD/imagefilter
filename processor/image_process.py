# tools/image_processor.py
import asyncio
import os
import logging
from io import BytesIO

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from PIL import Image
import imagehash

load_dotenv()


class ImageProcessor:
    """下載圖片並回傳尺寸 / 哈希等資訊；內建併發與速率控制"""

    def __init__(self) -> None:
   
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", 8))
        self._sema_download = asyncio.Semaphore(self.max_concurrent)
        self._new_conn_limiter = AsyncLimiter(2, 1)           # 每秒至多 2 次新連線


        self._connector_cfg = dict(
            limit=self.max_concurrent * 2,
            limit_per_host=8,
            keepalive_timeout=45,
            ttl_dns_cache=600,
        )
        self._connector: TCPConnector | None = None

    
        self._timeout = ClientTimeout(total=None, connect=10, sock_read=45)

  
    def _get_connector(self) -> TCPConnector:
        """第一次呼叫時才真正建立 TCPConnector；之後直接重用"""
        if self._connector is None:
            self._connector = TCPConnector(**self._connector_cfg)
        return self._connector


    async def _fetch_image(
        self, session: aiohttp.ClientSession, url: str, retries: int = 4, backoff: float = 0.7
    ) -> bytes | None:
        """帶重試與限速的 GET；成功回傳 bytes，失敗回 None"""
        for attempt in range(retries):
            try:
                async with self._new_conn_limiter:          #  新建連線速率
                    async with self._sema_download:         #  同時下載上限
                        async with session.get(url, timeout=self._timeout) as resp:
                            if resp.status == 200:
                                return await resp.read()
                            if resp.status == 404:
                                logging.warning("404  Not-Found: %s", url)
                                return None                 # 永久失敗
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.error("下載失敗 %s | %s | %s", url, e.__class__.__name__, repr(e))

            await asyncio.sleep(backoff * (2 ** attempt))   # 0.7,1.4,2.8,5.6
        return None


    async def _analyze_bytes(self, raw: bytes, url: str, ext: str) -> dict:
        """Pillow 解碼 + pHash；若解碼錯誤會回傳 None 欄位"""
        try:
            with Image.open(BytesIO(raw)) as img:
                w, h = img.size
                phash = str(imagehash.phash(img))
                aspect_ratio = round(max(w, h) / min(w, h), 3) if h else None
                dpi = img.info.get("dpi", (None,))[0]
        except Exception as e:
            logging.error("解碼失敗 %s | %s", url, repr(e))
            return {
                "url": url,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "dpi": None,
                "extension": ext,
                "phash": None,
                "raw": raw,
            }

        return {
            "url": url,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "dpi": dpi,
            "extension": ext,
            "phash": phash,
            "raw": raw,
        }


    async def process_image(self, session: aiohttp.ClientSession, url: str) -> dict:
        ext = os.path.splitext(url)[1].lower()
        raw = await self._fetch_image(session, url)
        if raw is None:
            return {
                "url": url,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "dpi": None,
                "extension": ext,
                "phash": None,
                "raw": None,
            }
        return await self._analyze_bytes(raw, url, ext)

    # 批次下載介面：保留給 CSV 前處理用
    async def process_images(self, urls: list[str]) -> list[dict]:
        results: list[dict] = []
        async with aiohttp.ClientSession(connector=self._get_connector()) as session:

            async def worker(u: str):
                async with self._sema_download:  # 保險
                    return await self.process_image(session, u)

            tasks = [asyncio.create_task(worker(u)) for u in urls]
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
        return results


