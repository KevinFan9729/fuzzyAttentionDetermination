import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

speech_engine = pyttsx3.init()
voices = speech_engine.getProperty('voices')
speech_engine.setProperty('voice', voices[3].id)


font = cv2.FONT_HERSHEY_SIMPLEX 

LEFT_INEX=[474,475,476,477]
RIGHT_INEX=[469,470,471,472]

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

prompt_show=200
initial_calibrate=0
center_calibrated=0
left_calibrated=0
right_calibrated=0

x_list=[]
y_list=[]
x_center=0
y_center=0

x_left_list=[]
y_left_list=[]
x_left=0
y_left=0

x_right_list=[]
y_right_list=[]
x_right=0
y_right=0

speak_once=0


def speak(text):
  speech_engine.say(text)
  speech_engine.runAndWait()

def calibrateGaze(mode,x_right_iris,y_right_iris):
  global x_center, y_center, x_left, y_left, x_right, y_right
  if mode=="center":
    text="please look at the center"
    if speak_once==0:
      threading.Thread(target=speak, args=(text,), daemon=True).start()
    # cv2.putText(image,text,(50,100), font, 0.8,(0,0,255),1,cv2.LINE_AA)
    x_list.append(x_right_iris)
    y_list.append(y_right_iris)
    if len(x_list)>10:#discard the data from the initial 10 frame due to possible transition contamination 
      x_center=sum(x_list[9:])/len(x_list[9:])
      y_center=sum(y_list[9:])/len(y_list[9:])
  elif mode=="left":
    text="please look left"
    if speak_once==0:
      threading.Thread(target=speak, args=(text,), daemon=True).start()
    # cv2.putText(image,text,(50,100), font, 0.8,(0,0,255),1,cv2.LINE_AA)
    x_list.append(x_right_iris)
    y_list.append(y_right_iris)
    if len(x_list)>10:
      x_left=sum(x_list[9:])/len(x_list[9:])
      y_left=sum(y_list[9:])/len(y_list[9:])
  elif mode=="right":
    text="please look right"
    if speak_once==0:
      threading.Thread(target=speak, args=(text,), daemon=True).start()
    # cv2.putText(image,text,(50,100), font, 0.8,(0,0,255),1,cv2.LINE_AA)
    x_list.append(x_right_iris)
    y_list.append(y_right_iris)
    if len(x_list)>10:
      x_right=sum(x_list[9:])/len(x_list[9:])
      y_right=sum(y_list[9:])/len(y_list[9:])

def gazeDirection(x_right_iris):
  #logic for checking gaze direction
  global x_center, y_center, x_left, y_left, x_right, y_right
  tolerance_left=abs(x_left-x_center)*0.5
  tolerance_right=abs(x_center-x_right)*0.5
  if (x_center-tolerance_right) <=x_right_iris<= (x_center+tolerance_left):
    direction="center"
  elif x_right_iris > (x_center+tolerance_left):
    direction="left"
  elif x_right_iris < (x_center-tolerance_right):
    direction="right"
  print(direction)
    # print(x_right_iris)
# def draw_circle(event,x,y,flags,param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(image,(x,y),100,(255,0,0),-1)
#         mouseX,mouseY = x,y


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
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_h,image_w=image.shape[:2]
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # print(face_landmarks.landmark)
        normalized_xy=np.array([([p.x,p.y]) for p in face_landmarks.landmark])
        points_xy=np.multiply(normalized_xy,[image_w,image_h]).astype(np.int32)
        # print(normalized_xy.shape)
        left_iris=points_xy[LEFT_INEX]
        right_iris=points_xy[RIGHT_INEX]
        # print(left_iris)
        # cv2.polylines(image,[left_iris],True,(0,0,255),1,cv2.LINE_AA)
        # cv2.polylines(image,[right_iris],True,(0,0,255),1,cv2.LINE_AA)

        (x_left_iris, y_left_iris), radius_left = cv2.minEnclosingCircle(left_iris)
        (x_right_iris, y_right_iris), radius_right = cv2.minEnclosingCircle(right_iris)
        center_left = np.array((x_left_iris, y_left_iris),dtype=np.int32)
        center_right = np.array((x_right_iris, y_right_iris),dtype=np.int32)
        radius_left, radius_right = int(radius_left), int(radius_right)
        cv2.circle(image,center_left,1,(0,0,255),1,cv2.LINE_AA)
        cv2.circle(image,center_right,1,(0,0,255),1,cv2.LINE_AA)
        # print(x_right_iris)
        # print(y_right_iris)
        if center_calibrated==1 and left_calibrated==1 and right_calibrated==1:
          text="calibration complete, thank you"
          initial_calibrate=1
          if speak_once==0:
            threading.Thread(target=speak, args=(text,), daemon=True).start()
            speak_once=1
        if initial_calibrate==0:
          if center_calibrated==0:
            calibrateGaze("center",x_right_iris,y_right_iris)
            speak_once=1
            prompt_show-=1
            if prompt_show==0:
              print("center")
              center_calibrated=1
              prompt_show=200
              speak_once=0
              x_list.clear()
              y_list.clear()
              print((x_center, y_center))
          elif left_calibrated==0:
            calibrateGaze("left",x_right_iris,y_right_iris)
            prompt_show-=1
            speak_once=1
            if prompt_show==0:
              print("left")
              left_calibrated=1
              prompt_show=200
              speak_once=0
              x_list.clear()
              y_list.clear()
              print((x_left, y_left))
          elif right_calibrated==0:
            calibrateGaze("right",x_right_iris,y_right_iris)
            prompt_show-=1
            speak_once=1
            if prompt_show==0:
              print("right")
              right_calibrated=1
              prompt_show=200
              speak_once=0
              x_list.clear()
              y_list.clear()
              print((x_right, y_right))
        else:
          gazeDirection(x_right_iris)
# center
# (300.07364227135145, 235.9112622325957)
# left
# (368.2436342888478, 233.9476401443881)
# right
# (262.4741819691284, 234.6271339935782)

    image=cv2.flip(image, 1)
    # cv2.setMouseCallback('image',draw_circle)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == 27:#escpae key
      break
    # elif ord('a'):
    #     print (mouseX,mouseY)

cap.release()
