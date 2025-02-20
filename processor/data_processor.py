import asyncio
import json
import os
import aiohttp
import imagehash
from tools.csv_processor import CSVProcessor
from tools.image_processor import ImageProcessor

class DataProcessor:
    def __init__(self, csv_file: str = None, result_file: str = None):
        """
        初始化 DataProcessor：
          - 若提供 csv_file 與 result_file，則支援 CSV 數據處理；
          - 否則主要支援處理使用者 POST 輸入的資料。
        """
        self.csv_file = csv_file
        self.result_file = result_file
        self.image_processor = ImageProcessor()

    async def process_csv_and_images(self):
        """
        從 CSV 讀取結構化記錄，使用圖片 URL 進行處理，
        將原始記錄與圖片處理結果合併，
        並對同一飯店（key 相同）的記錄進行重複檢測（使用 pairwise 比較），
        返回格式：
        {
          "total_count": <記錄數>,
          "results": [
             { "row": <>, "key": <>, "url": <>, "tag": [
                 { "type": "width", "name": "xxx" },
                 { "type": "height", "name": "xxx" },
                 { "type": "ppi", "name": "xxx" },
                 { "type": "aspect_ratio", "name": "xxx" },
                 { "type": "repeat", "name": "<canonical_url>" }  // 如果檢測到重複
             ] },
             ...
          ]
        }
        """
        csv_processor = CSVProcessor()
        try:
            records = csv_processor.get_image_records()
        except Exception as e:
            raise Exception(f"讀取 CSV 失敗: {e}")
        total_count = len(records)
        if not records:
            raise Exception("CSV 中無有效圖片連結。")
        # 提取所有 URL 用於批量處理
        urls = [record["url"] for record in records]
        results = await self.image_processor.process_images(urls)
        # 合併 CSV 原始記錄與圖片處理結果，暫存 phash 用於重複檢測
        combined = []
        for rec, proc in zip(records, results):
            width = proc.get("width") if proc.get("width") is not None else 0
            height = proc.get("height") if proc.get("height") is not None else 0
            aspect_ratio = proc.get("aspect_ratio") if proc.get("aspect_ratio") is not None else 0
            ppi = width * height
            tags = [
                {"type": "width", "name": str(width)},
                {"type": "height", "name": str(height)},
                {"type": "ppi", "name": str(ppi)},
                {"type": "aspect_ratio", "name": str(aspect_ratio)}
            ]
            combined.append({
                "row": rec["row"],
                "key": rec["key"],
                "url": rec["url"],
                "tag": tags,
                "phash": proc.get("phash")  # 暫存 phash 用於重複檢測
            })
        # 對同一飯店（key 相同）的記錄進行重複檢測（逐對比較）
        combined = self._detect_duplicates(combined, threshold=5)
        # 刪除所有記錄中暫存的 phash 字段
        for rec in combined:
            rec.pop("phash", None)
        processed_data = {
            "total_count": total_count,
            "results": combined
        }
        if self.result_file:
            with open(self.result_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
        return processed_data

    def _detect_duplicates(self, records: list, threshold: int) -> list:
        """
        對同一飯店（相同 key）的記錄進行重複檢測，採用逐對比較方法：
        - 將記錄按 key 分組，然後對每個組內的記錄按 row 升序排序，
        - 逐對比較記錄的 phash（通過 imagehash 轉換後計算 Hamming 距離），
        如果兩筆記錄的 Hamming 距離 <= threshold，則認為這兩張圖片重複，
        並在雙方的 tag 列表中分別增加對方的 URL（即雙向標記）。
        
        返回經過重複檢測處理後的所有記錄列表。
        """
        # 將記錄按 key 分組
        groups = {}
        for rec in records:
            groups.setdefault(rec["key"], []).append(rec)

        # 對每個分組進行處理
        for key, recs in groups.items():
            # 按 row 升序排序，較小的 row 視為較早出現，方便閱讀
            recs.sort(key=lambda x: x["row"])
            n = len(recs)
            # 逐對比較：對於每個索引 i 的記錄，與索引 i+1 之後的記錄進行比較
            for i in range(n):
                rec_i = recs[i]
                phash_i = rec_i.get("phash")
                if not phash_i:
                    continue
                hash_i = imagehash.hex_to_hash(phash_i)
                for j in range(i + 1, n):
                    rec_j = recs[j]
                    phash_j = rec_j.get("phash")
                    if not phash_j:
                        continue
                    hash_j = imagehash.hex_to_hash(phash_j)
                    # 若 Hamming 距離小於等於 threshold，則認為兩張圖片重複，
                    # 並在雙方的 tag 列表中分別添加對方的 URL
                    if (hash_i - hash_j) <= threshold:
                        rec_i["tag"].append({"type": "repeat", "name": rec_j["url"]})
                        rec_j["tag"].append({"type": "repeat", "name": rec_i["url"]})
        # 將各組記錄合併成一個列表返回
        new_records = []
        for recs in groups.values():
            new_records.extend(recs)
        return new_records


    async def init_data_if_needed(self):
        """
        如果 RESULT_FILE 存在，則讀取；否則調用 process_csv_and_images 生成數據。
        """
        try:
            with open(self.result_file, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
        except (FileNotFoundError, TypeError):
            processed_data = await self.process_csv_and_images()
        return processed_data

    async def process_input_data(self, records: list) -> list:
        """
        處理使用者透過 POST 輸入的資料（列表，每筆記錄包含 row, key, url）。
        對每筆資料使用圖片 URL 進行異步處理，返回格式與 CSV 處理結果一致，
        並進行重複檢測：
          - 對同一飯店（key 相同）的記錄按 row 排序，
          - 對於每一筆，與後續記錄比較 Hamming 距離，
          - 若 Hamming 距離 <= threshold，則在重複記錄的 tag 中新增 repeat 標籤。
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._process_single_record(rec, session) for rec in records]
            results = await asyncio.gather(*tasks)
        # 按 key 分組並檢測重複
        results = self._detect_duplicates(results, threshold=5)
        # 刪除暫存的 phash 字段
        for rec in results:
            rec.pop("phash", None)
        return results

    async def _process_single_record(self, rec, session) -> dict:
        row = rec.row if hasattr(rec, "row") else rec.get("row")
        key = rec.key if hasattr(rec, "key") else rec.get("key")
        url = rec.url if hasattr(rec, "url") else rec.get("url", "")
        if not url:
            return {"row": row, "key": key, "url": url, "tag": [], "phash": ""}
        result = await self.image_processor.process_image(session, url)
        if result.get("width") is None and result.get("height") is None and result.get("aspect_ratio") is None:
            tags = []
        else:
            width = result.get("width") if result.get("width") is not None else 0
            height = result.get("height") if result.get("height") is not None else 0
            aspect_ratio = result.get("aspect_ratio") if result.get("aspect_ratio") is not None else 0
            ppi = width * height
            tags = [
                {"type": "width", "name": str(width)},
                {"type": "height", "name": str(height)},
                {"type": "ppi", "name": str(ppi)},
                {"type": "aspect_ratio", "name": str(aspect_ratio)}
            ]
        return {"row": row, "key": key, "url": url, "tag": tags, "phash": result.get("phash", "")}
