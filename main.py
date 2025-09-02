import os
import cv2
import time
from collections import deque
from input.camera_stream import capture_frames
from detector.yolo_detector import detect_from_frame
from detector.i3d_detector import run_prediction
from detector.severity_selector import select_severity
from database.event_logger import log_event
from llm.llm_summary import generate_summary_from_events
from reports.report_generator import generate_pdf_report
from alerts.telegram_bot import send_alert, send_pdf_with_summary  # Telegram integration

# -------------------- Config --------------------
# VIDEO_SOURCE = r"./tests/V_19.mp4"  # or 0 for webcam
VIDEO_SOURCE = 0  # or 0 for webcam

CLIP_LEN = 40                       # I3D clip length
SCREENSHOT_DIR = r"./output/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

ALERT_COOLDOWN = 15  # seconds between consecutive danger alerts

event_buffer = []  # Collect events for LLM summary
last_alert_time = 0

# -------------------- Main --------------------
def main():
    global last_alert_time
    print("[INFO] Starting Mini SentryAI+...")

    for frame, clip in capture_frames(VIDEO_SOURCE):

        # -------------------- YOLO detection --------------------
        yolo_results = detect_from_frame(frame)

        # -------------------- I3D detection --------------------
        i3d_pred = "normal"
        if len(clip) >= CLIP_LEN:
            clip_preprocessed = [cv2.resize(f, (128,128)) for f in clip]
            i3d_pred, conf = run_prediction(clip_preprocessed)

        
        # -------------------- Severity selection --------------------
        severity = select_severity(yolo_results, i3d_pred)

        # -------------------- Event logging --------------------
        log_event(frame, yolo_results, i3d_pred, severity)

        # -------------------- Save annotated screenshot --------------------
        save_screenshot = severity in ["danger", "suspicious"]
        screenshot_path = None

        if save_screenshot:
            frame_id = len(event_buffer) + 1
            annotated_frame = frame.copy()
            for det in yolo_results:
                x1, y1, x2, y2 = map(int, det["bbox"])
                color = (0,0,255) if det["severity"]=="danger" else (0,255,255)
                label = f"{det['class']} ({det['severity']}) {det['confidence']:.2f}"
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            severity_color = (0,0,255) if severity=="danger" else (0,255,255)
            cv2.putText(annotated_frame, f"Final Severity: {severity}", (35,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, severity_color, 2)

            screenshot_path = os.path.join(SCREENSHOT_DIR, f"event_{frame_id}.jpg")
            cv2.imwrite(screenshot_path, annotated_frame)

            # Add to event buffer
            event_buffer.append({
                "frame": frame_id,
                "yolo": ", ".join([det['class'] for det in yolo_results]),
                "i3d": i3d_pred,
                "final": severity,
                "screenshot": screenshot_path
            })

            # -------------------- Telegram alert with cooldown --------------------
            if severity == "danger" and (time.time() - last_alert_time) > ALERT_COOLDOWN:
                send_alert(annotated_frame, severity)
                last_alert_time = time.time()

        # -------------------- Visualization --------------------
        for det in yolo_results:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = (0,0,255) if det["severity"]=="danger" else (0,255,255)
            label = f"{det['class']} ({det['severity']}) {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        severity_color = (0,0,255) if severity=="danger" else (0,255,255) if severity=="suspicious" else (0,255,0)
        cv2.putText(frame, f"Final Severity: {severity}", (35,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, severity_color, 2)

        cv2.imshow("Mini SentryAI+", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # -------------------- Generate LLM summary --------------------
    if event_buffer:
        summary_text = generate_summary_from_events(event_buffer)
        print("[INFO] LLM Summary Generated:")
        print(summary_text)

        # -------------------- Generate PDF report --------------------
        pdf_path = os.path.join("./output", "MiniSentryAI_Report.pdf")
        generate_pdf_report(event_buffer, summary_text, pdf_path)
        print(f"[INFO] PDF report saved to {pdf_path}")

        # -------------------- Send final PDF via Telegram --------------------
        send_pdf_with_summary(pdf_path, summary_text)

if __name__ == "__main__":
    main()

