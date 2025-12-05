from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "grain_physical.pt"
model = YOLO(str(MODEL_PATH))

def analyze_image(image_path: str) -> dict:
    results = model.predict(image_path, conf=0.25)
    
    # Count detections by class
    whole_count = 0
    broken_count = 0
    foreign_count = 0
    discolored_count = 0
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:
                whole_count += 1
            elif class_id == 1:
                broken_count += 1
            elif class_id == 2:
                foreign_count += 1
            elif class_id == 3:
                discolored_count += 1
    
    total_grains = whole_count + broken_count + discolored_count
    
    if total_grains == 0:
        return {
            "error": "No grains detected",
            "grade": None
        }
    
    head_rice_pct = (whole_count / total_grains) * 100
    broken_pct = (broken_count / total_grains) * 100
    discolored_pct = (discolored_count / total_grains) * 100
    
    grade = "Substandard"
    
    if (head_rice_pct >= 95.0 and 
        discolored_pct <= 0.5 and 
        foreign_count == 0):
        grade = "Premium"
    elif (head_rice_pct >= 80.0 and 
          discolored_pct <= 2.0 and 
          foreign_count <= 1):
        grade = "Grade 1"
    elif (head_rice_pct >= 65.0 and 
          discolored_pct <= 4.0):
        grade = "Grade 2"
    elif (head_rice_pct >= 50.0 and 
          discolored_pct <= 8.0):
        grade = "Grade 3"
    
    return {
        "grade": grade,
        "headRicePercent": str(round(head_rice_pct, 1)),
        "brokenPercent": str(round(broken_pct, 1)),
        "discoloredPercent": str(round(discolored_pct, 1)),
        "foreignObjects": foreign_count,
        "totalGrains": total_grains,
        "counts": {
            "whole": whole_count,
            "broken": broken_count,
            "discolored": discolored_count,
            "foreign": foreign_count
        }
    }