import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class CSVProcessor:
    def __init__(self):
       
        self.csv_file = os.getenv("CSV_FILE")
        print("正在使用的 CSV 檔案路徑：", self.csv_file)

    def get_image_urls(self):
        try:
            with open(self.csv_file, mode='r', encoding='utf-8', errors='replace') as f:
                df = pd.read_csv(f)
        except Exception as e:
            raise Exception(f"讀取 CSV 檔案失敗：{e}")

        if "RoomImgPath" not in df.columns:
            raise ValueError("CSV 檔案中找不到 'RoomImgPath' 欄位。")

        urls = df["RoomImgPath"].dropna().tolist()
        return urls
