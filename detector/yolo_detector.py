from ultralytics import YOLO

# -------------------- Config --------------------
DANGER_CLASSES = ['fire', 'gun']
SUSPICIOUS_CLASSES = ['mask', 'helmet', 'knife']
NORMAL_CLASSES = ['person']

MODEL_PATH = "./models/best.pt"   # adjust path if needed
model = YOLO(MODEL_PATH)

# -------------------- Helpers --------------------
def classify_detection(cls_name):
    """Map YOLO class name into severity category."""
    if cls_name in DANGER_CLASSES:
        return "danger"
    elif cls_name in SUSPICIOUS_CLASSES:
        return "suspicious"
    elif cls_name in NORMAL_CLASSES:
        return "normal"
    else:
        return "unknown"

def detect_from_frame(frame):
    """
    Run YOLO detection on a single frame (NumPy array).
    Returns a list of detections: 
    [{'class':..., 'severity':..., 'confidence':..., 'bbox':[x1,y1,x2,y2]}, ...]
    """
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            severity = classify_detection(cls_name)
            detections.append({
                "class": cls_name,
                "severity": severity,
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

    return detections

# Optional: for testing with image path
def detect_from_image(image_path):
    import cv2
    frame = cv2.imread(image_path)
    return detect_from_frame(frame)

