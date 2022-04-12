import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


cap=cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)#mirror image
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w=
        results=face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            print(results.multi_face_landmarks[0].landmark)
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()