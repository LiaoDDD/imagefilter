import os

# 測試環境變數是否正確載入
print("CSV_FILE =", os.environ.get("CSV_FILE"))
print("RESULT_FILE =", os.environ.get("RESULT_FILE"))
print("INPUT_CACHE_FILE =", os.environ.get("INPUT_CACHE_FILE"))
print("SCENE_MODEL_WEIGHT =", os.environ.get("SCENE_MODEL_WEIGHT"))
print("SCENE_LABEL_FILE =", os.environ.get("SCENE_LABEL_FILE"))
print("LIGHTING_MODEL_PATH =", os.environ.get("LIGHTING_MODEL_PATH"))
print("LIGHTING_LABEL_FILE =", os.environ.get("LIGHTING_LABEL_FILE"))
