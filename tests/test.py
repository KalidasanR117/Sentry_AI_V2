from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Test on an image
# results = model("test.jpg", show=True)

# Or test on webcam
results = model.predict(source=0, show=True)  # 0 = default webcam
