from input.camera_stream import CameraStream
from input.frame_extractor import FrameExtractor
from detector.yolo_detector import YOLODetector
import cv2

def main():
    cam = CameraStream(source=0)   # 0 = webcam
    extractor = FrameExtractor(resize=(640, 480))
    detector = YOLODetector(model_path="yolov8n.pt", conf=0.5)

    print("[INFO] Press 'q' to quit.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            print("[INFO] End of stream or cannot access camera.")
            break

        processed = extractor.preprocess(frame)
        detections = detector.detect(processed)
        annotated = detector.draw_detections(processed.copy(), detections)

        cv2.imshow("SentryAI+ YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()

if __name__ == "__main__":
    main()
