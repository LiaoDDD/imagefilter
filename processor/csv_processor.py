import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class CSVProcessor:
    def __init__(self):
        self.csv_file = os.getenv("CSV_FILE")
        print("正在使用的 CSV 檔案路徑：", self.csv_file)

    def get_image_urls(self):
        """
        返回 CSV 中的圖片 URL 列表（僅使用 'RoomImgPath' 欄位）。
        """
        try:
            with open(self.csv_file, mode='r', encoding='utf-8', errors='replace') as f:
                df = pd.read_csv(f)
        except Exception as e:
            raise Exception(f"讀取 CSV 檔案失敗：{e}")
        if "RoomImgPath" not in df.columns:
            raise ValueError("CSV 檔案中找不到 'RoomImgPath' 欄位。")
        return df["RoomImgPath"].dropna().tolist()

    def get_image_records(self):
        """
        返回結構化圖片記錄列表，每筆記錄包含 'row', 'key' 與 'url'。
        假設 CSV 中有 'RoomImgPath' 與 'hotelid' 欄位；若無 'hotelid' 則統一設為 "unknown"，
        且 'row' 為 CSV 行號（從 1 開始）。
        """
        try:
            with open(self.csv_file, mode='r', encoding='utf-8', errors='replace') as f:
                df = pd.read_csv(f)
        except Exception as e:
            raise Exception(f"讀取 CSV 檔案失敗：{e}")
        if "RoomImgPath" not in df.columns:
            raise ValueError("CSV 檔案中找不到 'RoomImgPath' 欄位。")
        if "hotelid" not in df.columns:
            df["hotelid"] = "unknown"
        records = []
        for index, row in df.iterrows():
            if pd.notnull(row["RoomImgPath"]):
                record = {
                    "row": index + 1,
                    "key": row["hotelid"],
                    "url": row["RoomImgPath"]
                }
                records.append(record)
        return records
