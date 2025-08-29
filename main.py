from input.camera_stream import CameraStream
from input.frame_extractor import FrameExtractor
import cv2

def main():
    cam = CameraStream(source=0)   # 0 = default webcam, or put "video.mp4"
    extractor = FrameExtractor(resize=(640, 480))

    print("[INFO] Press 'q' to quit.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            print("[INFO] End of stream or cannot access camera.")
            break

        processed = extractor.preprocess(frame)

        cv2.imshow("SentryAI+ Camera", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()

if __name__ == "__main__":
    main()
