import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

speech_engine = pyttsx3.init()
voices = speech_engine.getProperty('voices')
speech_engine.setProperty('voice', voices[3].id)


font = cv2.FONT_HERSHEY_SIMPLEX 

# 1 nose tip
# 33 left eye corner
# 263 right eye corner
# 61 left mouth corner
# 291 right mouth corner
# 199 chin
# 168 between eye brows
# 104 left forehead
# 334 right forehead
# 151 forehead

# FACIAL_KEYS=[33,263,1,61,291,199]
FACIAL_KEYS=[33, 263, 168, 105, 334, 151]
# NOSE=[1]
FOREHEAD= [151]

# For webcam input:
cap = cv2.VideoCapture(0)

def getHeadOrientation(x,y):
    
    if y < -12:
        headOrient = "Head Left"
    elif y > 12:
        headOrient = "Head Right"
    elif x < -20:
        headOrient = "Head Down"
    elif x > -3:
        headOrient = "Head Up"
    else:
        headOrient = "Head Center"
    return headOrient


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,#include iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.flip(image, 1)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_h,image_w=image.shape[:2]
    facial_keypoints_2d = []

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # print(face_landmarks.landmark)
        normalized_xyz=np.array([([p.x,p.y,p.z]) for p in face_landmarks.landmark])
        points_xyz=np.multiply(normalized_xyz,[image_w, image_h, 1]).astype(np.float32)
        facial_keypoints_3d=points_xyz[FACIAL_KEYS]
        for i in range(len(facial_keypoints_3d)):
            facial_keypoints_2d.append(facial_keypoints_3d[i][0:2])
        # nose_tip_3d=points_xyz[NOSE]
        # nose_tip_3d[0][2]=nose_tip_3d[0][2]*1
        # nose_tip_2d=nose_tip_3d[0][0:2]
        forehead_3d=points_xyz[FOREHEAD]
        forehead_3d[0][2]=forehead_3d[0][2]*1
        forehead_2d=forehead_3d[0][0:2]

        facial_keypoints_2d=np.array(facial_keypoints_2d)

        # print(facial_keypoints_2d)

        # The camera matrix
        focal_length = image_w
        cam_center = (image_w/2, image_h/2)
        cam_matrix = np.array([ [focal_length, 0, cam_center[0]],
                                [0, focal_length, cam_center[1]],
                                [0, 0, 1]],dtype=np.float32)

        dist_coeffs = np.zeros((4,1)) # Assume no lens distortion
        success, rot_vec, trans_vec = cv2.solvePnP(facial_keypoints_3d, facial_keypoints_2d, cam_matrix, dist_coeffs)
        
        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # print(rmat)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # tx=math.atan2(rmat[2][1],rmat[2][2])*(180/math.pi)
        # ty=math.atan2(-1*rmat[2][0],math.sqrt(rmat[2][1]**2+rmat[2][2]**2))*(180/math.pi)
        # tz=math.atan2(rmat[1][0],rmat[0][0])*(180/math.pi)

        # print((tx,ty,tz))
        # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
        
        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        text=getHeadOrientation(x,y)

        # # Display the nose direction
        # nose_3d_projection, _ = cv2.projectPoints(nose_tip_3d, rot_vec, trans_vec, cam_matrix, dist_coeffs)

        # p1 = (int(nose_tip_2d[0]), int(nose_tip_2d[1]))
        # p2 = (int(nose_tip_2d[0] + y * 10) , int(nose_tip_2d[1] - x * 10))

        # Display the nose direction
        forehead_projection, _ = cv2.projectPoints(forehead_3d, rot_vec, trans_vec, cam_matrix, dist_coeffs)

        p1 = (int(forehead_2d[0]), int(forehead_2d[1]))
        p2 = (int(forehead_2d[0] + y * 10) , int(forehead_2d[1] - x * 10))
        
        cv2.line(image, p1, p2, (255, 0, 0), 1,cv2.LINE_AA)

        # Add the text on the image
        cv2.putText(image, text, (20, 50), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), font, 1, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), font, 1, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), font, 1, (0, 0, 255), 2,cv2.LINE_AA)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == 27:#escpae key
      break

cap.release()

    