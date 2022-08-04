import cv2
import mediapipe as mp
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


font = cv2.FONT_HERSHEY_SIMPLEX 

RIGHT_INEX=[474,475,476,477]
LEFT_INEX=[469,470,471,472]

# FACIAL_KEYS=[33,263,1,61,291,199]
FACIAL_KEYS=[33, 263, 168, 105, 334, 151]

NOSE=[1]
FOREHEAD= [151]

LEFT_EYE_KEYS= [133,33] #right key, left key
RIGHT_EYE_KEYS = [263,362] #right key, left key



#increase contrast
#increase brightness
#historgram equilazation

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

prompt_show=200
initial_calibrate=0
center_calibrated=0
left_calibrated=0
right_calibrated=0
recalibrate_first_time=0
recal_flag=0

recalibration=[]
past=0 # history calibration data

x_center=0 # data collected from the right eye 
x2_center=0 # data collected from the left eye

x_left_list=[]
x2_left_list=[]
x_left=0
x2_left=0

x_right_list=[]
x2_right_list=[]
x_right=0
x2_right=0


def fuzzyEye(iris_pos, iris2_pos, headDir):
  if headDir == "Head Left":
    eyePos = ctrl.Antecedent(np.arange(x_left+5, (x_right+5+0.05), 0.05), 'eye position')
    x_fuzzy_center=x_center+5
    x_fuzzy_left=x_left+5
    x_fuzzy_right=x_right+5+0.05

    eyePos2 = ctrl.Antecedent(np.arange(x2_left+5, (x2_right+5+0.05), 0.05), 'left eye position')
    x2_fuzzy_center=x2_center+5
    x2_fuzzy_left=x2_left+5
    x2_fuzzy_right=x2_right+5+0.05

  elif headDir == "Head Right":
    eyePos = ctrl.Antecedent(np.arange(x_left+2, (x_right+10+0.05), 0.05), 'eye position')
    x_fuzzy_center=x_center+2
    x_fuzzy_left=x_left+2
    x_fuzzy_right=x_right+2+0.05

    eyePos2 = ctrl.Antecedent(np.arange(x2_left+2, (x2_right+10+0.05), 0.05), 'left eye position')
    x2_fuzzy_center=x2_center+2
    x2_fuzzy_left=x2_left+2
    x2_fuzzy_right=x2_right+2+0.05

  else: # head center
    eyePos = ctrl.Antecedent(np.arange(x_left, (x_right+0.05), 0.05), 'eye position')
    x_fuzzy_center=x_center
    x_fuzzy_left=x_left
    x_fuzzy_right=x_right+0.05

    eyePos2 = ctrl.Antecedent(np.arange(x2_left, (x2_right+0.05), 0.05), 'left eye position')
    x2_fuzzy_center=x2_center
    x2_fuzzy_left=x2_left
    x2_fuzzy_right=x2_right+0.05

  
  eyeDir = ctrl.Consequent(np.arange(0, 1.05, 0.05), 'eye direction')


  #right eye position membership function
  # eyePos['low'] = fuzz.trimf(eyePos.universe, [x_fuzzy_left-10, x_fuzzy_left-10, x_fuzzy_center])#left point, middle point, rigt point
  # eyePos['med'] = fuzz.trimf(eyePos.universe, [x_fuzzy_left+int((x_fuzzy_center-x_fuzzy_left-10)/2), x_fuzzy_center, x_fuzzy_right+10-int((x_fuzzy_right+10-x_fuzzy_center)/2)])
  # eyePos['high'] = fuzz.trimf(eyePos.universe, [x_fuzzy_center, x_fuzzy_right+10, x_fuzzy_right+10])
  
  
  eyePos['low'] = fuzz.gaussmf(eyePos.universe, x_fuzzy_left, 8)
  eyePos['med'] = fuzz.gaussmf(eyePos.universe, x_fuzzy_center,8)
  eyePos['high'] = fuzz.gaussmf(eyePos.universe, x_fuzzy_right,8)
  
  #left eye position membership function
  # eyePos2['low'] = fuzz.trimf(eyePos2.universe, [x2_fuzzy_left-10, x2_fuzzy_left-10, x2_fuzzy_center])#left point, middle point, rigt point
  # eyePos2['med'] = fuzz.trimf(eyePos2.universe, [x2_fuzzy_left+int((x2_fuzzy_center-x2_fuzzy_left-10)/2), x2_fuzzy_center, x2_fuzzy_right+10-int((x2_fuzzy_right+10-x2_fuzzy_center)/2)])
  # eyePos2['high'] = fuzz.trimf(eyePos2.universe, [x2_fuzzy_center, x2_fuzzy_right+10, x2_fuzzy_right+10])

  eyePos2['low'] = fuzz.gaussmf(eyePos2.universe, x2_fuzzy_left, 8)
  eyePos2['med'] = fuzz.gaussmf(eyePos2.universe, x2_fuzzy_center,8)
  eyePos2['high'] = fuzz.gaussmf(eyePos2.universe, x2_fuzzy_right,8)
  
  #eye direction level membership function
  # eyeDir['left'] = fuzz.trimf(eyeDir.universe, [0, 0, 0.5])#left point, middle point, rigt point
  # eyeDir['center'] = fuzz.trimf(eyeDir.universe, [0.25,0.5,0.75])
  # eyeDir['right'] = fuzz.trimf(eyeDir.universe, [0.5, 1, 1])
  
  eyeDir['left'] = fuzz.gaussmf(eyeDir.universe, 0, 0.1)
  eyeDir['center'] = fuzz.gaussmf(eyeDir.universe, 0.5, 0.1)
  eyeDir['right'] = fuzz.gaussmf(eyeDir.universe, 1, 0.1)

  #fuzzy rules
  rule1 = ctrl.Rule((eyePos['low'] | eyePos2['low']),eyeDir['left'])
  rule2 = ctrl.Rule((eyePos['med'] | eyePos2['med']),eyeDir['center'])
  rule3 = ctrl.Rule((eyePos2['high'] | eyePos['high']),eyeDir['right'])

  eyeDirCtrl = ctrl.ControlSystem([rule1, rule2, rule3])
  eyeLvl = ctrl.ControlSystemSimulation(eyeDirCtrl)



  eyeLvl.input['eye position'] = iris_pos
  eyeLvl.input['left eye position']=iris2_pos
  # eyePos.view()
  # eyePos2.view()
  # eyeDir.view()
  eyeLvl.compute()
  # a=eyeLvl.output['eye direction']

  # print(a)
  eye_output=eyeLvl.output['eye direction']
  if headDir =="Head Center":
    if 0.4<=eye_output<=0.55:
        gazeDir="Eye Contact"
        print("eye contact")
        print(eyeLvl.output['eye direction'])
    else:
        gazeDir="Eye Away"
        print("Eye Away")
        print(eyeLvl.output['eye direction'])
    return gazeDir
  elif headDir =="Head Right":
    if 0.35<=eye_output<=0.45:
        gazeDir="Eye Contact"
        print("eye contact")
        print(eyeLvl.output['eye direction'])
    else:
        gazeDir="Eye Away"
        print("Eye Away")
        print(eyeLvl.output['eye direction'])
    return gazeDir
  else:
    if 0.38<=eye_output<=0.45:
        gazeDir="Eye Contact"
        print("eye contact")
        print(eyeLvl.output['eye direction'])
    else:
        gazeDir="Eye Away"
        print("Eye Away")
        print(eyeLvl.output['eye direction'])
    return gazeDir
      
  # eyePos.view(sim=eyeLvl)
  # eyeDir.view(sim=eyeLvl)

def attentionDecsion(headDir, eyeDir):
  if headDir == "Head Center" and eyeDir == "Eye Contact":
    attention = "full attention"
  elif headDir == "Head Center" and eyeDir == "Eye Away":
    attention = "semi-attention"
  elif headDir == "Head Left" and eyeDir == "Eye Contact":
    attention = "semi-attention"
  elif headDir == "Head Left" and eyeDir == "Eye Away":
    attention = "no attention"
  elif headDir == "Head Right" and eyeDir == "Eye Contact":
    attention = "semi-attention"
  elif headDir == "Head Right" and eyeDir == "Eye Away":
    attention = "no attention"
  elif headDir == "Head Down":
    attention = "no attention"
  elif headDir == "Head Up":
    attention = "semi-attention"
  else:
    attention = "no attention"
  return attention



def gazeDirection(x_iris):
  #logic for checking gaze direction
  global x_center, y_center, x_left, y_left, x_right, y_right
  tolerance_left=abs(x_center-x_left)*0.5
  tolerance_right=abs(x_right-x_center)*0.5
  if (x_center-tolerance_left) <=x_iris<= (x_center+tolerance_right):
    direction="Eye Center"
  elif x_iris < (x_center-tolerance_left):
    direction="Eye Left"
  elif x_iris > (x_center+tolerance_right):
    direction="Eye Right"
  print(direction)
  return direction


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
def preporcess(image):
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limag = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limag, cv2.COLOR_Lab2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    final=cv2.flip(final, 1)
    return final



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
    image=preporcess(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image=cv2.flip(image, 1)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_h,image_w=image.shape[:2]
    facial_keypoints_2d = []
    left_iris=[]
    right_iris=[]
    left_eye_keys_2d = []
    right_eye_keys_2d = []


    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # print(face_landmarks.landmark)
        # normalized_xy=np.array([([p.x,p.y]) for p in face_landmarks.landmark])
        # points_xy=np.multiply(normalized_xy,[image_w,image_h]).astype(np.int32)

        normalized_xyz=np.array([([p.x,p.y,p.z]) for p in face_landmarks.landmark])
        points_xyz=np.multiply(normalized_xyz,[image_w, image_h, 1]).astype(np.float32)

        facial_keypoints_3d=points_xyz[FACIAL_KEYS]
        for i in range(len(facial_keypoints_3d)):
            facial_keypoints_2d.append(facial_keypoints_3d[i][0:2])
        
        for i in range(2):
            left_eye_keys_2d.append(points_xyz[LEFT_EYE_KEYS][i][0:2])
            right_eye_keys_2d.append(points_xyz[RIGHT_EYE_KEYS][i][0:2])

        forehead_3d=points_xyz[FOREHEAD]
        forehead_3d[0][2]=forehead_3d[0][2]*1
        forehead_2d=forehead_3d[0][0:2]

        facial_keypoints_2d=np.array(facial_keypoints_2d)

        iris_left_3d=points_xyz[LEFT_INEX]
        iris_right_3d=points_xyz[RIGHT_INEX]

        for i in range(len(iris_left_3d)):
            left_iris.append(iris_left_3d[i][0:2])
            right_iris.append(iris_right_3d[i][0:2])
        
        left_iris = np.array(left_iris, dtype=np.int32)
        right_iris = np.array(right_iris, dtype=np.int32)

        # The camera matrix
        focal_length = image_w
        cam_center = (image_w/2, image_h/2)
        cam_matrix = np.array([ [focal_length, 0, cam_center[0]],
                                [0, focal_length, cam_center[1]],
                                [0, 0, 1]],dtype=np.float32)

        # Assume no lens distortion
        dist_coeffs = np.zeros((4,1))
        success, rot_vec, trans_vec = cv2.solvePnP(facial_keypoints_3d, facial_keypoints_2d, cam_matrix, dist_coeffs) #SOLVEPNP_ITERATIVE

        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        (x_left_iris, y_left_iris), radius_left = cv2.minEnclosingCircle(left_iris)
        (x_right_iris, y_right_iris), radius_right = cv2.minEnclosingCircle(right_iris)
        center_left = np.array((x_left_iris, y_left_iris),dtype=np.int32)
        center_right = np.array((x_right_iris, y_right_iris),dtype=np.int32)
        radius_left, radius_right = int(radius_left), int(radius_right)
        cv2.circle(image,center_left,1,(0,0,255),1,cv2.LINE_AA)
        cv2.circle(image,center_right,1,(0,0,255),1,cv2.LINE_AA)

        x_center=(right_eye_keys_2d[0][0]+right_eye_keys_2d[1][0])/2
        x2_center=(left_eye_keys_2d[0][0]+left_eye_keys_2d[1][0])/2
        x_left=right_eye_keys_2d[1][0]
        x2_left=left_eye_keys_2d[1][0]
        x_right=right_eye_keys_2d[0][0]
        x2_right=left_eye_keys_2d[0][0]

        # plot referencing keypoints for gaze direction
        # x_center=(x_left+x_right)/2
        # x2_center=(x2_left+x2_right)/2
        
        # test1 = np.array(left_eye_keys_2d[0],dtype=np.int32)
        # test1[0]-=10
        # test2 = np.array(left_eye_keys_2d[1],dtype=np.int32)
        # test3 = np.array(right_eye_keys_2d[0],dtype=np.int32)
        # test4 = np.array(right_eye_keys_2d[1],dtype=np.int32)
        # test4[0]+=10

        # cv2.circle(image,test1,1,(255,0,255),1,cv2.LINE_AA)
        # cv2.circle(image,test2,1,(255,0,255),1,cv2.LINE_AA)

        # cv2.circle(image,test3,1,(0,255,255),1,cv2.LINE_AA)
        # cv2.circle(image,test4,1,(0,255,255),1,cv2.LINE_AA)

        # cv2.circle(image,(int(x_center),test3[1]),1,(0,255,255),1,cv2.LINE_AA)
        # cv2.circle(image,(int(x2_center),test2[1]),1,(0,255,255),1,cv2.LINE_AA)
   

        headDir=getHeadOrientation(x,y)
        eyeDir=fuzzyEye(x_right_iris, x_left_iris, headDir)
        # print(x_left_iris,x_right_iris)
        attention=attentionDecsion(headDir,eyeDir)
        # recal_flag=1
        # # recalibrate(forehead_2d[0])
        # print(attention)
        # # print(x_right_iris)
        
        p1 = (int(forehead_2d[0]), int(forehead_2d[1]))
        p2 = (int(forehead_2d[0] + y * 5) , int(forehead_2d[1] + x * 5))
        
        cv2.line(image, p1, p2, (255, 0, 0), 1,cv2.LINE_AA)
        
        # Add the text on the image
        cv2.putText(image, headDir, (20, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if eyeDir=="Eye Away":
          cv2.putText(image, eyeDir, (20, 100), font, 1, (0, 0, 255), 2,cv2.LINE_AA)
        else:
          cv2.putText(image, eyeDir, (20, 100), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
        
        if attention=="full attention":
          cv2.putText(image, attention, (420, 50), font, 1, (0, 255, 0), 2,cv2.LINE_AA)
        elif attention == "semi-attention":
          cv2.putText(image, attention, (400, 50), font, 1, (255, 0, 0), 2,cv2.LINE_AA)
        else:
          cv2.putText(image, attention, (420, 50), font, 1, (0, 0, 255), 2,cv2.LINE_AA)
        # cv2.putText(image, eyeDir, (500, 50), font, 1, (0, 0, 255), 2,cv2.LINE_AA)

    cv2.imshow('head and gaze', image)
    if cv2.waitKey(1) & 0xFF == 27:#escpae key
      break
   

cap.release()

