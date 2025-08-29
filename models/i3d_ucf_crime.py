import torch
import torch.nn as nn
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import time

# ------------------------------
# Model definition
# ------------------------------
class I3DClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.i3d = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
        self.dropout = nn.Dropout(p=0.3)
        self.i3d.blocks[6].proj = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.dropout(self.i3d(x))


# ------------------------------
# Load pretrained I3D (fine-tuned on UCF-Crime)
# ------------------------------
def load_i3d_ucf_finetuned():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = I3DClassifier().to(device)

    weights_path = hf_hub_download(
        repo_id="Ahmeddawood0001/i3d_ucf_finetuned",
        filename="i3d_ucf_finetuned.pth"
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


# ------------------------------
# Frame extraction
# ------------------------------
def extract_frames_from_list(frames, max_frames=32, frame_size=(224, 224)):
    if len(frames) == 0:
        raise ValueError("⚠️ No frames captured!")

    processed = []
    for f in frames[:max_frames]:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, frame_size)
        processed.append(f)

    while len(processed) < max_frames:
        processed.append(processed[-1])

    arr = np.stack(processed)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
    return tensor.unsqueeze(0)


# ------------------------------
# Classification
# ------------------------------
def classify_frames(frames, model, labels=None):
    if labels is None:
        labels = ["Arrest", "Explosion", "Fight", "Normal",
                  "RoadAccidents", "Shooting", "Stealing", "Vandalism"]

    device = next(model.parameters()).device
    tensor = extract_frames_from_list(frames).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = int(probs.argmax(dim=1))
        return labels[idx], float(probs[0, idx].cpu())


# ------------------------------
# Webcam mode
# ------------------------------
def run_webcam(model, interval=2, max_frames=32):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Could not open webcam.")
        return

    labels = ["Arrest", "Explosion", "Fight", "Normal",
              "RoadAccidents", "Shooting", "Stealing", "Vandalism"]

    last_time = time.time()
    frame_buffer = []
    prediction = "..."
    confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        # Run classification every `interval` seconds
        if time.time() - last_time >= interval:
            try:
                prediction, confidence = classify_frames(frame_buffer, model, labels)
            except Exception as e:
                prediction, confidence = "Error", 0.0
                print("Classification error:", e)

            frame_buffer = []
            last_time = time.time()

        # Draw prediction
        cv2.putText(frame, f"{prediction} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2)

        cv2.imshow("I3D UCF-Crime - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    model = load_i3d_ucf_finetuned()

    # # ✅ Option 1: Run on video file
    # video_path = r"H:\Sentry_AI_V2\tests\124.mp4"
    # try:
    #     frames = [cv2.imread(video_path)]  # dummy check
    #     label, conf = classify_frames([cv2.imread(video_path)], model)
    #     print(f"Predicted: {label} (confidence {conf:.2f})")
    # except Exception as e:
    #     print("File test failed:", e)

    # ✅ Option 2: Run on webcam
    print("🎥 Starting webcam... press 'q' to quit")
    run_webcam(model)
