from pathlib import Path
import json

TXT_PATH   = Path("/app/test/urls_data.txt")
JSON_PATH  = Path("/app/test/test_data2.json")
KEY_PREFIX = "480+11899886+LC00001254+314193595+"

def convert(txt_path: Path, json_path: Path, key_prefix: str) -> None:
    """讀取 txt -> 產生 JSON -> 寫檔"""
    data = []
    with txt_path.open(encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            url = line.strip()
            if not url:            
                continue
            data.append({
                "row": idx,
                "key": f"{key_prefix}{idx + 1}", 
                "url": url
            })

    json_path.write_text(
        json.dumps({"Data": data}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ 已寫入 {json_path}，共 {len(data)} 筆資料")

if __name__ == "__main__":
    convert(TXT_PATH, JSON_PATH, KEY_PREFIX)
