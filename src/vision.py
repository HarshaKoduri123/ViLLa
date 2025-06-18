import cv2
from collections import defaultdict
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

def detect_animals(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    animal_data = []
    animal_counts = defaultdict(int)

    for box in results.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id].lower()
        bbox = box.xyxy[0].tolist()

        animal_data.append({
            "class": class_name,
            "bbox": [round(coord, 2) for coord in bbox]
        })
        animal_counts[class_name] += 1

    return animal_data, dict(animal_counts)
