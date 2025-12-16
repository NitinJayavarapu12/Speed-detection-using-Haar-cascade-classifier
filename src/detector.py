import cv2

class VehicleDetector:
    def __init__(self, cascade_path):
        self.cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, gray_image):
        return self.cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40)
        )

