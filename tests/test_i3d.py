import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# --- Use relative paths for better portability ---
# Assumes 'models' folder is in the same directory as your script
MODEL_PATH = "models/keras_model.h5" 
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

# 1. Load pretrained model
print("Loading model...")
model = load_model(MODEL_PATH)

# 2. Parameters (from training setup)
IMG_SIZE = 224
SEQ_LEN = 16  # Number of frames the model expects
CLASS_LABELS = ["Non-Violence", "Violence"] # Define labels based on model training

# 3. Extract frames from video
def load_video(path, seq_len=SEQ_LEN):
    """
    Loads and preprocesses a video from a given path.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {path}")
        return None
        
    frames = []
    while len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            # If video ends early, pad with the last frame
            if not frames:
                print("Error: Video is empty or unreadable.")
                return None
            while len(frames) < seq_len:
                frames.append(frames[-1])
            break
            
        # --- CRITICAL: Convert from BGR (OpenCV) to RGB (Keras/TensorFlow) ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized_frame = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        normalized_frame = resized_frame.astype("float32") / 255.0
        frames.append(normalized_frame)
    
    cap.release()
    return np.array(frames)

# 4. Run inference
video_path = "tests/124.mp4" # Your test video (using a relative path)
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

print(f"Processing video: {video_path}")
clip = load_video(video_path)

if clip is not None:
    # Add the batch dimension: (16, 224, 224, 3) -> (1, 16, 224, 224, 3)
    input_clip = np.expand_dims(clip, axis=0)

    # Predict
    pred = model.predict(input_clip, verbose=0)
    
    # --- Use argmax for a more robust result ---
    pred_class_index = np.argmax(pred[0])
    pred_class_label = CLASS_LABELS[pred_class_index]
    confidence = pred[0][pred_class_index] * 100

    print(f"Raw prediction array: {pred[0]}")
    
    if pred_class_label == "Violence":
        print(f"Result: {pred_class_label} ⚠️ (Confidence: {confidence:.2f}%)")
    else:
        print(f"Result: {pred_class_label} 🚶 (Confidence: {confidence:.2f}%)")