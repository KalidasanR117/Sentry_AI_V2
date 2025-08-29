from ultralytics import YOLO
import cv2

# Class categories
DANGER_CLASSES = ['fire', 'gun']
SUSPICIOUS_CLASSES = ['mask', 'helmet', 'knife']
NORMAL_CLASSES = ['person']

# Load model
MODEL_PATH = "models/best.pt"   # adjust path if needed
model = YOLO(MODEL_PATH)

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

def detect_from_image(image_path, show=False):
    """Run detection on an image."""
    results = model(image_path, show=show)
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

def detect_from_webcam(source=0):
    """Run YOLO detection on webcam feed."""
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                severity = classify_detection(cls_name)

                # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{cls_name} ({severity}) {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Quick test
    # detections = detect_from_image("test.jpg", show=True)
    # print(detections)
    detect_from_webcam()   # Uncomment to test webcam
