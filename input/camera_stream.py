import cv2

class CameraStream:
    def __init__(self, source=0):
        """
        Initialize the camera stream.
        :param source: 0 = default webcam, or path to a video file.
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

    def read_frame(self):
        """
        Read a single frame from the camera/video.
        :return: frame (BGR) or None if end of stream.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the camera/video capture."""
        self.cap.release()
        cv2.destroyAllWindows()
