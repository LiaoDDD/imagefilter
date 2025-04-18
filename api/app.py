import os
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from tools.data_processor import DataProcessor
from dotenv import load_dotenv
from utils.api import API
import time

load_dotenv()  # 載入 .env 檔案中的設定

app = FastAPI(title="CSV Image Analysis API", root_path="/imagefilter")

# 從環境變數中獲取 CSV 與結果檔案路徑（CSV 方式使用）
CSV_FILE = os.getenv("CSV_FILE")
RESULT_FILE = os.getenv("RESULT_FILE")
INPUT_CACHE_FILE = os.getenv("INPUT_CACHE_FILE")
SCENE_MODEL_WEIGHT = os.getenv("SCENE_MODEL_WEIGHT")
SCENE_LABEL_FILE = os.getenv("SCENE_LABEL_FILE")
LIGHTING_MODEL_PATH = os.getenv("LIGHTING_MODEL_PATH")
LIGHTING_LABEL_FILE = os.getenv("LIGHTING_LABEL_FILE")

# 初始化 DataProcessor 實例
data_processor = DataProcessor(
    csv_file=CSV_FILE,
    result_file=RESULT_FILE,
    input_cache_file=INPUT_CACHE_FILE,
    scene_model_weight=SCENE_MODEL_WEIGHT,
    scene_label_file=SCENE_LABEL_FILE,
    lighting_model_path=LIGHTING_MODEL_PATH,
    lighting_label_file=LIGHTING_LABEL_FILE
                )

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
      - 判斷使否有csv處裡
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
    從 CSV 結果檔案讀取處理後的圖片資料並回傳
    返回格式統一為：
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
    response = API.result_format(data, r_desc, r_code)
    return JSONResponse(content=response, status_code=200)

@app.post("/analyze_images")
async def analyze_images(input_data: InputData):
    start_time = time.perf_counter()
    """
    POST 接口：接收使用者提交的 JSON 數據座位input格式需要正確。
    對input進行處裡
      1. 讀取緩存 (INPUT_CACHE_FILE) ，判斷每筆資料是否已處理過。
         若已存在，則直接取用該筆資料中的 tag
      2. 未處理過的則進行圖片處理並更新緩存。
      3. 合併所有資料後，執行重複檢測，對重複圖片雙向補充標籤，
         每筆重複記錄均列出其他重複圖片的 URL。
      4. 刪除暫存的 phash 欄位後返回，且保持輸入順序與 row 不變。
    回傳格式與 GET 接口一致。
    """
    if not input_data.Data:
        raise HTTPException(status_code=400, detail="未提供有效的 Data。")
    
    results = await data_processor.process_input_data(input_data.Data)
    data = {"results": results}
    r_desc, r_code = API.get_r_desc_r_code(data)

    distinct_hotel_ids = set()
    for rec in input_data.Data:
        key = rec.key
        hotel_id = key.split("+")[0] if key and "+" in key else key
        distinct_hotel_ids.add(hotel_id)
    
    # 如果有超過一個不同的 hotel id，則狀態碼設為 201
    status_code = 306 if len(distinct_hotel_ids) > 1 else 200

    response = API.result_format(data, r_desc, r_code)
    end_time = time.perf_counter()
    total_time_sec = end_time - start_time
    print(f"Total processing time: {total_time_sec:.2f} seconds")
    return JSONResponse(content=response, status_code=status_code)

