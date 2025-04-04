import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('landmark_model.h5')

def preprocess(frame):
    face_gray = cv2.resize(frame, (96, 96)) / 255.0
    return face_gray.reshape(1, 96, 96, 1)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (96, 96))

    input_img = preprocess(face)
    preds = model.predict(input_img)[0]
    preds = preds * 96  # scale back to image size

    for i in range(0, len(preds), 2):
        x = int(preds[i])
        y = int(preds[i+1])
        cv2.circle(frame, (x + 200, y + 100), 2, (0, 255, 0), -1)  # adjust offset as needed

    cv2.imshow("Landmark Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
