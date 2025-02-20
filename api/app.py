import asyncio
import os
import json
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from tools.data_processor import DataProcessor
from dotenv import load_dotenv
from utils.api import API

load_dotenv()  # 載入 .env 檔案中的設定

app = FastAPI(title="CSV Image Analysis API", root_path="/imagefilter")

# 從環境變數中獲取 CSV 與結果檔案路徑（CSV 方式使用）
CSV_FILE = os.getenv("CSV_FILE")
RESULT_FILE = os.getenv("RESULT_FILE")

# 初始化 DataProcessor 實例
data_processor = DataProcessor(csv_file=CSV_FILE, result_file=RESULT_FILE)

# 定義 POST 輸入資料模型
class InputRecord(BaseModel):
    row: int
    key: str
    url: str

class InputData(BaseModel):
    Data: List[InputRecord]

@app.on_event("startup")
async def startup_event():
    """
    應用啟動時：
      - 若 RESULT_FILE 存在則讀取（CSV 處理結果），
      - 否則通過 CSV 處理圖片數據生成結果並存檔。
    """
    try:
        await data_processor.init_data_if_needed()
    except Exception as e:
        print(f"初始化資料失敗: {e}")

@app.get("/analyze_csv")
async def analyze_csv(
    max_results: int = Query(1000, ge=1, description="每次返回的最大筆數"),
    continuation_token: int = Query(0, ge=0, description="查詢索引 (起始筆數)")
):
    """
    GET 接口：返回 CSV 處理後的圖片數據。
    返回格式統一為：
    {
      "results": [
         { "row": <>, "key": <>, "url": <>, "tag": [ ... ] },
         ...
      ]
    }
    """
    try:
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取結果檔失敗: {e}")
    
    all_results = processed_data.get("results", [])
    if not all_results:
        raise HTTPException(status_code=400, detail="無有效資料。")
    
    data = {"results": all_results}
    r_desc, r_code = API.get_r_desc_r_code(data)
    return API.result_format(data, r_desc, r_code)

@app.post("/analyze_images")
async def analyze_images(input_data: InputData):
    """
    POST 接口：接收使用者提交的 JSON 數據，格式例如：
    {
      "Data": [
         { "row": 1, "key": "htl0001twin001", "url": "https://..." },
         { "row": 2, "key": "htl0001twin001", "url": "https://..." }
      ]
    }
    系統將對每條記錄透過 URL 下載並處理圖片，提取 width、height、ppi（width×height）及 aspect_ratio，
    並針對相同 key（代表同一飯店）的記錄進行重複檢測，對於檢測到的重複圖片，在其 tag 列表中新增一個
    { "type": "repeat", "name": "<canonical_url>" }。返回格式與 CSV 處理接口保持一致：
    {
      "results": [
         { "row": <>, "key": <>, "url": <>, "tag": [ ... ] },
         ...
      ]
    }
    """
    if not input_data.Data:
        raise HTTPException(status_code=400, detail="未提供有效的 Data。")
    
    results = await data_processor.process_input_data(input_data.Data)
    data = {"results": results}
    r_desc, r_code = API.get_r_desc_r_code(data)
    return API.result_format(data, r_desc, r_code)
