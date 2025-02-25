import asyncio
import json
import os
import aiohttp
import imagehash
import logging
from tools.csv_processor import CSVProcessor
from tools.image_processor import ImageProcessor

logging.basicConfig(level=logging.INFO)

class DataProcessor:
    def __init__(self, csv_file: str = None, result_file: str = None, input_cache_file: str = None):
        """
        初始化
        """
        self.csv_file = csv_file
        self.result_file = result_file
        self.input_cache_file = input_cache_file
        self.image_processor = ImageProcessor()

    async def process_csv_and_images(self):
        """
        從 CSV 中讀取記錄並處理圖片數據，
        合併 CSV 記錄與圖片屬性（暫存 phash 用於重複檢測），
        執行重複檢測後將結果獨立保存至 result_file，並返回處理結果。
        """
        csv_processor = CSVProcessor()
        try:
            records = csv_processor.get_image_records()
        except Exception as e:
            logging.error(f"讀取 CSV 失敗: {e}")
            raise Exception(f"讀取 CSV 失敗: {e}")
        total_count = len(records)
        if not records:
            raise Exception("CSV 中無有效圖片連結。")
        
        # 處理 CSV 記錄中的圖片 URL
        urls = [record["url"] for record in records]
        results = await self.image_processor.process_images(urls)
        
        # 合併 CSV 記錄與圖片處理結果，暫存 phash 用於重複檢測
        combined = []
        for rec, proc in zip(records, results):
            if (proc.get("width") is None and proc.get("height") is None and proc.get("aspect_ratio") is None):
                tags = []
            else:
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
                "phash": proc.get("phash")  # 暫存欄位，用於重複檢測
            })
        
        # 執行重複檢測（逐對比較），不修改原始 row
        combined = self._detect_duplicates(combined, threshold=5)
        for rec in combined:
            rec.pop("phash", None)
        
        processed_data = {
            "total_count": total_count,
            "results": combined
        }
        if self.result_file:
            try:
                with open(self.result_file, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error(f"寫入 CSV 結果檔案失敗: {e}")
        return processed_data

    def _detect_duplicates(self, records: list, threshold: int) -> list:
        """
        對圖片進行比對
          1. 將記錄依 key 分組（僅用於比對，不修改原始 row）。
          2. 比對phash比對phash
             若距離 <= threshold 則認為兩筆記錄重複
             並在雙向 tag 列表中添加對方的 URL。
        返回處理後的記錄列表，順序保持與原輸入一致。
        """
        groups = {}
        for rec in records:
            groups.setdefault(rec["key"], []).append(rec)
        
        for key, recs in groups.items():
            # 這裡建立排序用的複本，不修改原始記錄
            sorted_recs = sorted(recs, key=lambda x: x["row"])
            n = len(sorted_recs)
            for i in range(n):
                rec_i = sorted_recs[i]
                phash_i = rec_i.get("phash")
                if not phash_i:
                    continue
                try:
                    hash_i = imagehash.hex_to_hash(phash_i)
                except Exception as e:
                    logging.error(f"轉換 phash 失敗: {phash_i}, 錯誤: {e}")
                    continue
                for j in range(i + 1, n):
                    rec_j = sorted_recs[j]
                    phash_j = rec_j.get("phash")
                    if not phash_j:
                        continue
                    try:
                        hash_j = imagehash.hex_to_hash(phash_j)
                    except Exception as e:
                        logging.error(f"轉換 phash 失敗: {phash_j}, 錯誤: {e}")
                        continue
                    if (hash_i - hash_j) <= threshold:
                        # 雙向添加重複標籤
                        if not any(tag["name"] == rec_j["url"] for tag in rec_i["tag"]):
                            rec_i["tag"].append({"type": "duplicate", "name": rec_j["url"]})
                        if not any(tag["name"] == rec_i["url"] for tag in rec_j["tag"]):
                            rec_j["tag"].append({"type": "duplicate", "name": rec_i["url"]})
        new_records = []
        for recs in groups.values():
            new_records.extend(recs)
        # 保持原始輸入順序（依 row 排序，因為 input 的 row 不會變動）
        new_records.sort(key=lambda x: x["row"])
        return new_records

    def _load_cache(self) -> dict:
        """
        從 input_cache_file 中載入緩存，回傳以 URL 為鍵的字典。
        若檔案不存在或解析失敗，則回傳空字典。
        """
        if not self.input_cache_file:
            return {}
        try:
            with open(self.input_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cache = {}
            for rec in data.get("results", []):
                url = rec.get("url")
                if url:
                    cache[url] = rec
            return cache
        except Exception as e:
            logging.warning(f"載入緩存失敗: {e}")
            return {}

    def _save_cache(self, cache: dict):
        """
        將緩存字典保存到 input_cache_file 中，格式：
        """
        if not self.input_cache_file:
            return
        records = []
        for rec in cache.values():
            # 複製一份記錄，排除 row 欄位
            new_rec = {k: v for k, v in rec.items() if k != "row"}
            records.append(new_rec)
        data = {
            "total_count": len(records),
            "results": records
        }
        try:
            with open(self.input_cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.error(f"保存緩存失敗: {e}")

    async def process_input_data(self, records: list) -> list:
        """
        處理使用者透過 POST 輸入的input。
        流程：
          1. 從 input_cache_file 載入已處理結果（以 url 為鍵）
          2. 遍歷每筆輸入記錄：
             - 若緩存中已存在，則直接採用該記錄。
             - 否則進行圖片處理並更新緩存。
          3. 將所有記錄合併後統一執行Duplicste。
          4. 刪除暫存的 phash 欄位後返回最終結果給使用者。
        """
        cache = self._load_cache()
        new_results = []
        async with aiohttp.ClientSession() as session:
            to_process = []
            for rec in records:
                input_row = rec.row if hasattr(rec, "row") else rec.get("row")
                input_key = rec.key if hasattr(rec, "key") else rec.get("key")
                url = rec.url if hasattr(rec, "url") else rec.get("url", "")
                if url in cache:
                    cached = cache[url]
                    new_record = {
                        "row": input_row,
                        "key": input_key,
                        "url": url,
                        "tag": cached.get("tag", []),
                        "phash": cached.get("phash")
                    }
                    new_results.append(new_record)
                else:
                    to_process.append(rec)
            if to_process:
                tasks = [self._process_single_record(rec, session) for rec in to_process]
                processed = await asyncio.gather(*tasks)
                for idx, rec in enumerate(to_process):
                    input_row = rec.row if hasattr(rec, "row") else rec.get("row")
                    input_key = rec.key if hasattr(rec, "key") else rec.get("key")
                    processed[idx]["row"] = input_row
                    processed[idx]["key"] = input_key
                new_results.extend(processed)
                for rec in processed:
                    url = rec.get("url")
                    if url:
                        cache[url] = rec
                self._save_cache(cache)
        new_results = self._detect_duplicates(new_results, threshold=5)
        for rec in new_results:
            rec.pop("phash", None)
        # 保持輸入順序（依原始 row 排序）
        new_results.sort(key=lambda x: x["row"])
        return new_results

    async def _process_single_record(self, rec, session) -> dict:
        """
        處理單筆記錄，支援 rec 為 Pydantic 模型或字典。
        若 url 為空則直接返回空 tag
        否則呼叫 image_processor.process_image() 取得圖片屬性，
        計算 ppi 構造 tag 列表，並返回暫存 phash 供重複檢測使用。
        """
        row = rec.row if hasattr(rec, "row") else rec.get("row")
        key = rec.key if hasattr(rec, "key") else rec.get("key")
        url = rec.url if hasattr(rec, "url") else rec.get("url", "")
        if not url:
            return {"row": row, "key": key, "url": url, "tag": [], "phash": ""}
        result = await self.image_processor.process_image(session, url)
        if (result.get("width") is None and 
            result.get("height") is None and 
            result.get("aspect_ratio") is None):
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
