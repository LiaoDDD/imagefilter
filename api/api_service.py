from fastapi import FastAPI, HTTPException, Query
from utils.csv_processor import CSVProcessor
from utils.image_processor import ImageProcessor

app = FastAPI(title="CSV Image Analysis API")

@app.get("/analyze_csv")
async def analyze_csv(
    max_results: int = Query(1000, ge=1, description="max"),
    continuation_token: int = Query(0, ge=0, description="查詢索引")
    ):
    csv_processor = CSVProcessor()
    try:
        urls = csv_processor.get_image_urls()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    total_count = len(urls)
    if not urls:
        raise HTTPException(status_code=400, detail="csv 中無有效圖片連結。")
    start = continuation_token
    end = continuation_token + max_results
    selected_urls = urls[start:end]
    
    processor = ImageProcessor()
    results = await processor.process_images(selected_urls)
    success_count = len(results)
    next_token = end if end < total_count else None
    return {
        "total_count": total_count,
        "max_results": max_results,
        "本業起始筆數": continuation_token,
        "下頁起始筆數": next_token,
        "total_success": success_count,
        "results": results
        }
