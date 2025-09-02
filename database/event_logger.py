import os
import json
from datetime import datetime
from pathlib import Path
import cv2

LOG_DIR = "./database/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "event_log.json")

def log_event(frame, yolo_results, i3d_pred, severity):
    """
    Log events to JSON file.
    frame: current frame (optional: save image separately)
    yolo_results: list of YOLO detections
    i3d_pred: prediction from I3D
    severity: final severity ("danger", "suspicious", "normal")
    """
    event = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "yolo_results": yolo_results,
        "i3d_prediction": i3d_pred
    }

    # -------------------- Load existing logs safely --------------------
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, ValueError):
            # If JSON is corrupted, reset logs
            logs = []

    logs.append(event)

    # -------------------- Save logs --------------------
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    # -------------------- Optional: save frame snapshot --------------------
    if severity == "danger":
        SNAP_DIR = Path(LOG_DIR) / "snapshots"
        SNAP_DIR.mkdir(exist_ok=True)
        filename = SNAP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(filename), frame)
