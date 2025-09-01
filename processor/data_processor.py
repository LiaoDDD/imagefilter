import asyncio
import json
import aiohttp
import imagehash
import logging
import numpy as np
import cv2
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from processor.csv_processor import CSVProcessor
from processor.image_processor import ImageProcessor
from processor.object_processor import ObjectProcessor
from processor.exposure_processor import ExposureProcessor
from processor.scene_processor import SceneProcessor
from processor.lighting_processor import LightingProcessor
main_logger = logging.getLogger("svc")
logging.basicConfig(level=logging.INFO)
CHUNK_SIZE = 150

class DataProcessor:
    def __init__(self, csv_file: str = None, result_file: str = None, input_cache_file: str = None, scene_model_weight: str = None, scene_label_file: str = None,lighting_model_path: str = None, lighting_label_file: str = None):
        """
        初始化
        """
        self.csv_file = csv_file
        self.result_file = result_file
        self.input_cache_file = input_cache_file
        self.decode_executor = ThreadPoolExecutor(max_workers=os.cpu_count()*4)
        self.stats = Counter()
        self.lock  = asyncio.Lock()
        max_workers = os.cpu_count() * 4
        self.decode_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.image_processor = ImageProcessor()
        self.object_processor = ObjectProcessor()
        self.exposure_processor = ExposureProcessor()
        self.lighting_processor = LightingProcessor(lighting_model_path, lighting_label_file) if lighting_model_path and lighting_label_file else None
        target_scenes = {
            "hotel_room",
            "bathroom"
        }
        if scene_model_weight and scene_label_file:
            self.scene_processor = SceneProcessor(scene_model_weight, scene_label_file, target_scenes=target_scenes)
        else:
            self.scene_processor = None

    async def init_data_if_needed(self):
        try:
            with open(self.result_file, "r", encoding="utf-8") as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # CSV 結果不存在或解析失敗，則重新處理 CSV 資料
            await self.process_csv_and_images()

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
        
        # 處理 CSV 中的圖片 URL
        urls = [record["url"] for record in records]
        results = await self.image_processor.process_images(urls)
        
        # 合併 CSV 與圖片處理結果，暫存 phash 用於重複檢測
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
        
        # 執行重複檢測，維持input row 一致
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
          1. 將記錄依 key 分組僅用於比對。
          2. 比對phash比對phash
             若距離 <= threshold 則認為兩筆記錄重複
             並在雙向 tag 列表中添加對方的 URL。
        返回處理後的記錄列表，順序保持與原輸入一致。
        """
        groups = {}
        for rec in records:
            hotel_id = rec["key"].split("+")[0] if rec.get("key") else ""
            groups.setdefault(hotel_id, []).append(rec)

        
        for hotel_id, recs in groups.items():
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
        從 input_cache_file 中載入緩存回傳以url為指標的字典。
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
        batch_start = time.perf_counter()


        async with self.lock:
            self.stats["total"] += len(records)

        cache = self._load_cache()
        new_results, to_process = [], []

        for rec in records:
            row = rec.row if hasattr(rec, "row") else rec["row"]
            key = rec.key if hasattr(rec, "key") else rec["key"]
            url = rec.url if hasattr(rec, "url") else rec["url"]
            if url in cache:
                cached = cache[url]
                new_results.append({**cached, "row": row, "key": key})
            else:
                to_process.append((row, key, url))

        # ── 2. 下載＋分析（分批）─────────────────────────────
        async with aiohttp.ClientSession(
            connector=self.image_processor._get_connector()) as session:

            for idx in range(0, len(to_process), CHUNK_SIZE):
                chunk = to_process[idx:idx+CHUNK_SIZE]
                tasks = [self._process_single_record(row, key, url, session)
                         for row, key, url in chunk]
                part = await asyncio.gather(*tasks)
                new_results.extend(part)

                # 更新 cache
                for rec in part:
                    cache[rec["url"]] = {k: v for k, v in rec.items() if k != "row"}

                self._save_cache(cache)
                main_logger.info("progress | %d/%d done", idx+len(chunk), len(to_process))

        # ── 3. Duplicate 檢測 & 移除暫存 phash
        dedup = self._detect_duplicates(new_results, threshold=5)
        for rec in dedup:
            rec.pop("phash", None)                
        dedup.sort(key=lambda x: x["row"])        

        # ── 4. 批次統計日誌
        elapsed = time.perf_counter() - batch_start
        async with self.lock:
            snap = dict(self.stats)
            self.stats.clear()

        main_logger.info(
            "batch_done | "
            f"recv={len(records)} "
            f"succ={snap.get('success', 0)} "
            f"dl_fail={snap.get('dl_fail', 0)} "
            f"cv_fail={snap.get('cv_fail', 0)} "
            f"yolo_fail={snap.get('yolo_fail', 0)} "
            f"task_fail={snap.get('task_fail', 0)} "
            f"elapsed={elapsed:.2f}s"
        )

        return dedup
    

    def _decode_and_analyze(self, raw: bytes):
        arr = np.frombuffer(raw, np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        exp    = self.exposure_processor.analyze(cv_img)
        return cv_img, exp
    
    # async def _process_single_record(self, rec, session) -> dict:
    #     """
    #     處理單筆記錄，支援 rec 為 Pydantic 模型或字典。
    #     若 url 為空則直接返回空 tag
    #     否則呼叫 image_processor.process_image() 取得圖片屬性，
    #     計算 ppi 構造 tag 列表，並返回暫存 phash 供重複檢測使用。
    #     """
    #     row = rec.row if hasattr(rec, "row") else rec.get("row")
    #     key = rec.key if hasattr(rec, "key") else rec.get("key")
    #     url = rec.url if hasattr(rec, "url") else rec.get("url", "")
    #     tags = []
    #     if not url:
    #         return {"row": row, "key": key, "url": url, "tag": [], "phash": ""}
        
    #     result = await self.image_processor.process_image(session, url)
    #     if result["raw"] is None:      # 下載或 Pillow 失敗直接早退
    #         async with self.lock: self.stats["dl_fail"] += 1
    #         return {"row": row, "key": key, "url": url,
    #                 "tag":[{"type":"download_error","name":"fail"}],
    #                 "phash": None}
    #     loop = asyncio.get_event_loop()
    #     try:
    #         cv_img, exposure = await loop.run_in_executor(
    #             self.decode_executor, self._decode_and_analyze, result["raw"])
    #     except Exception as e:
    #         main_logger.error(f"cv2_err | {url} | {repr(e)}")
    #         async with self.lock: self.stats["cv_fail"] += 1
    #         return {"row": row, "key": key, "url": url,
    #                 "tag":[{"type":"decode_error","name":"fail"}],
    #                 "phash": result["phash"]}
        
    #     w, h = result["width"], result["height"]
    #     tags += [
    #     {"type": "width",  "name": str(w)},
    #     {"type": "height", "name": str(h)},
    #     {"type": "ppi",    "name": str(w * h)},
    #     {"type": "aspect_ratio", "name": str(result["aspect_ratio"])}
    #         ]


    #     cv_image, exposure = None, None
    #     raw = result.get("raw")
    #     if raw:
    #         loop = asyncio.get_event_loop()
    #         try:
    #             cv_image, exposure = await loop.run_in_executor(
    #                 self.decode_executor, self._decode_and_analyze, raw)
    #         except Exception as e:
    #             main_logger.error(f"decode/exposure fail: {url} | {repr(e)}")
    #             async with self.lock: self.stats["cv_fail"] += 1
    #     detected_objects = []
    #     try:
    #         objs = self.object_processor.process(cv_img) or []
    #         tags += [{"type":"object","name":o} for o in objs]
    #     except Exception as e:
    #         main_logger.error(f"yolo_err | {url} | {repr(e)}")
    #         async with self.lock: self.stats["yolo_fail"] += 1
    #         tags.append({"type":"yolo_error","name":"fail"})

    #     custom_name = {"dining table": "table"}
    #     for obj in detected_objects:
    #         obj_tag = custom_name.get(obj, obj)
    #         tags.append({"type": "object", "name": obj_tag})

    #     if exposure:
    #         tags.append({"type":"exposure_status","name":exposure["exposure_status"]})
    #         tags.append({"type":"contrast_status","name":exposure["contrast_status"]})

    #     if self.scene_processor is not None:
    #         try:
    #             scene_result = self.scene_processor.analyze(url)
    #             if scene_result is not None:
    #                 tags.append({"type": "scene", "name": scene_result["scene"]})
    #         except Exception as e:
    #             main_logger.error(f"Scene fail: {url} | {repr(e)}")

    #     if self.lighting_processor is not None:
    #         try:
    #             lighting_result = self.lighting_processor.predict_from_cv2(cv_image)
    #             if lighting_result:
    #                 tags.append({"type": "light_source", "name": lighting_result})
    #         except Exception as e:
    #              main_logger.error(f"Light fail: {url} | {repr(e)}")
    #              async with self.lock: self.stats["light_fail"] += 1

    #     async with self.lock: self.stats["success"] += 1


    #     return {"row": row, "key": key, "url": url, "tag": tags, "phash": result.get("phash", "")}
# tools/data_processor.py 內，整個替換原函式

    async def _process_single_record(
        self,
        row: int,
        key: str,
        url: str,
        session: aiohttp.ClientSession
    ) -> dict:
        """
        處理單筆圖片紀錄：
        1. 若 url 為空，直接回空 tag
        2. 下載 + Pillow 解碼 (image_processor.process_image)
        3. OpenCV decode + Exposure (self._decode_and_analyze)
        4. YOLO 物件偵測、Exposure 標籤、Scene、Lighting
        5. 回傳包含 row/key/url/tag/phash 的 dict
        """
        # 1. 空 URL 早退
        if not url:
            return {"row": row, "key": key, "url": url, "tag": [], "phash": ""}

        # 2. 下載 & Pillow 解碼
        result = await self.image_processor.process_image(session, url)
        if result["raw"] is None:
            async with self.lock:
                self.stats["dl_fail"] += 1
            return {
                "row": row, "key": key, "url": url,
                "tag": [{"type": "download_error", "name": "fail"}],
                "phash": None
            }

        # 3. OpenCV 及 Exposure 分析
        loop = asyncio.get_running_loop()
        try:
            cv_img, exposure = await loop.run_in_executor(
                self.decode_executor,
                self._decode_and_analyze,
                result["raw"]
            )
        except Exception as e:
            main_logger.error("cv2_err | %s | %s", url, repr(e))
            async with self.lock:
                self.stats["cv_fail"] += 1
            return {
                "row": row, "key": key, "url": url,
                "tag": [{"type": "decode_error", "name": "fail"}],
                "phash": result["phash"]
            }

        # 4. 組基本 tags
        w, h = result["width"], result["height"]
        tags = [
            {"type": "width",           "name": str(w)},
            {"type": "height",          "name": str(h)},
            {"type": "ppi",             "name": str(w * h)},
            {"type": "aspect_ratio",    "name": str(result["aspect_ratio"])},
            {"type": "exposure_status", "name": exposure["exposure_status"]},
            {"type": "contrast_status", "name": exposure["contrast_status"]},
        ]

        # 5. YOLO 物件偵測
        try:
            objs = self.object_processor.process(cv_img) or []
            tags += [{"type": "object", "name": o} for o in objs]
        except Exception as e:
            main_logger.error("yolo_err | %s | %s", url, repr(e))
            async with self.lock:
                self.stats["yolo_fail"] += 1
            tags.append({"type": "yolo_error", "name": "fail"})

        # 6. Scene 分析 (若有啟用)
        if self.scene_processor:
            try:
                s_res = self.scene_processor.analyze(url)
                if s_res:
                    tags.append({"type": "scene", "name": s_res["scene"]})
            except Exception as e:
                main_logger.error("scene_err | %s | %s", url, repr(e))

        # 7. Lighting 分析 (若有啟用)
        if self.lighting_processor:
            try:
                light = self.lighting_processor.predict_from_cv2(cv_img)
                if light:
                    tags.append({"type": "light_source", "name": light})
            except Exception as e:
                main_logger.error("light_err | %s | %s", url, repr(e))
                async with self.lock:
                    self.stats["light_fail"] += 1

        # 8. 統計成功數
        async with self.lock:
            self.stats["success"] += 1

        # 9. 回傳結果
        return {
            "row": row,
            "key": key,
            "url": url,
            "tag": tags,
            "phash": result["phash"]
        }

