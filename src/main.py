import cv2
from detector import VehicleDetector
from speed_estimator import SpeedEstimator

VIDEO_PATH = "car.mp4"
CASCADE_PATH = "haarcascade_car.xml"
DISTANCE_METERS = 10

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

detector = VehicleDetector(CASCADE_PATH)
speed_estimator = SpeedEstimator(DISTANCE_METERS, fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    vehicles = detector.detect(gray)

    for idx, (x,y,w,h) in enumerate(vehicles):
        speed = speed_estimator.estimate(idx, x, w)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame,
                    f"{speed:.2f} km/h",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2)
        
        cv2.imshow("Vehicle Speed Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()