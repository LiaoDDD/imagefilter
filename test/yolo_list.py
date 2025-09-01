from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 取得類別名稱
class_names = model.names

print("YOLO 模型可辨識的物件類別如下：")
for class_id, class_name in class_names.items():
    print(f"{class_id:2}: {class_name}")
