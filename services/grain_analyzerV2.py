from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "grain_quality_detector.pt"

# Lazy load model to prevent memory issues at startup
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def analyze_image(image_path: str) -> dict:
    """
    Analyze a rice grain image and return NCT grading results.
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        Dictionary containing grain counts, percentages, and NCT grade
    """
    model = get_model()
    
    # Run Inference
    results = model.predict(image_path, conf=0.25, verbose=False)
    result = results[0]

    # Initialize Counters based on trained classes
    counts = {
        'Whole': 0,          # Head Rice
        'Broken': 0,
        'Chalky': 0,
        'Discolored': 0,     # Damaged/Discolored
        'Immature': 0,
        'Foreign Object': 0, # Organic/Inorganic Matter
        'Clean': 0           # Assuming 'Clean' maps to Whole/Good if present
    }

    # Count Detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Normalize class names to match counters if needed
        if class_name in counts:
            counts[class_name] += 1
        elif class_name == 'Damaged':  # Handle potential naming variations
            counts['Discolored'] += 1

    # Calculate Metrics (NCT Standards)
    # Total Grains = Whole + Broken + Chalky + Discolored + Immature
    # Foreign Objects are excluded from the denominator for grain percentages
    total_grains = (counts['Whole'] + counts['Broken'] + counts['Chalky'] +
                    counts['Discolored'] + counts['Immature'] + counts['Clean'])

    if total_grains > 0:
        # Calculate Percentages
        head_rice_pct = ((counts['Whole'] + counts['Clean']) / total_grains) * 100
        broken_pct = (counts['Broken'] / total_grains) * 100
        chalky_pct = (counts['Chalky'] / total_grains) * 100
        immature_pct = (counts['Immature'] / total_grains) * 100
        discolored_pct = (counts['Discolored'] / total_grains) * 100
    else:
        head_rice_pct = broken_pct = chalky_pct = immature_pct = discolored_pct = 0
        return {
            "error": "No grains detected",
            "grade": None,
            "totalGrains": 0
        }

    # Determine NCT Grade
    # Logic derived from NCT Manual Chapter XVI (Tables 2 & 4)
    grade = "FAIL / BELOW GRADE 3"  # Default

    # Evaluate from Highest (Premium) down to Lowest
    if (counts['Foreign Object'] == 0 and
        discolored_pct <= 0.5 and
        head_rice_pct >= 57.0 and
        chalky_pct < 2.0 and
        immature_pct < 2.0):
        grade = "PREMIUM"

    elif (counts['Foreign Object'] == 0 and
          discolored_pct <= 2.0 and
          head_rice_pct >= 48.0 and
          chalky_pct <= 5.0 and
          immature_pct <= 5.0):
        grade = "GRADE 1"

    elif (head_rice_pct >= 39.0 and
          chalky_pct <= 10.0 and
          immature_pct <= 10.0):
        grade = "GRADE 2"

    elif (head_rice_pct >= 30.0 and
          chalky_pct <= 15.0 and
          immature_pct <= 15.0):
        grade = "GRADE 3"

    return {
        "grade": grade,
        "headRicePercent": str(round(head_rice_pct, 2)),
        "brokenPercent": str(round(broken_pct, 2)),
        "chalkyPercent": str(round(chalky_pct, 2)),
        "immaturePercent": str(round(immature_pct, 2)),
        "discoloredPercent": str(round(discolored_pct, 2)),
        "foreignObjects": counts['Foreign Object'],
        "totalGrains": total_grains,
        "counts": {
            "whole": counts['Whole'],
            "broken": counts['Broken'],
            "chalky": counts['Chalky'],
            "immature": counts['Immature'],
            "discolored": counts['Discolored'],
            "clean": counts['Clean'],
            "foreignObject": counts['Foreign Object']
        }
    }