import cv2
import dlib
import time
import csv
from scipy.spatial import distance
import imutils

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# CSV output
csv_file = open("ear_data.csv", mode='w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["left_ear", "right_ear", "avg_ear", "label"])  # label: 1=drowsy, 0=alert

cap = cv2.VideoCapture(0)
print("Press 'a' to label ALERT, 'd' for DROWSY, 'q' to quit.")

time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [shape[i] for i in LEFT_EYE]
        right_eye = [shape[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        if avg_ear < 0.2:
            if time > 5:
                print("ASLEEP")
        else: 
            time = 0
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Collecting EAR Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):  # alert
        writer.writerow([left_ear, right_ear, avg_ear, 0])
        print("Logged: ALERT")
    elif key == ord('d'):  # drowsy
        writer.writerow([left_ear, right_ear, avg_ear, 1])
        print("Logged: DROWSY")
    elif key == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
