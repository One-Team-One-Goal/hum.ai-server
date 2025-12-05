from ultralytics import YOLO

model = YOLO("models/grain_physical.pt")

results = model("test_image/physical/test1.jpg", conf=0.25, save=True)

for result in results:
    print("Classes detected:", result.names)
    print("Boxes:", result.boxes)
    print("Number of detections:", len(result.boxes))
    
    # Print each detection
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"  - {class_name}: {confidence:.2%}")
