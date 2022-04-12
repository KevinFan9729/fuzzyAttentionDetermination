from chardet import detect
import cv2
import dlib
import numpy as np
from sympy import im



# void cv::pyrUp	(	InputArray 	src,
# OutputArray 	dst,
# const Size & 	dstsize = Size(),
# int 	borderType = BORDER_DEFAULT 
# )		
# Upsamples an image and then blurs it.

# try 5-point facial landmark detector
# look at mediapipe
# https://google.github.io/mediapipe/solutions/face_mesh.html
# https://github.com/google/mediapipe/issues/1530#issuecomment-819552656
# import mediapipe as mp
# import cv2
# mp_iris = mp.solutions.iris
# mp_draw = mp.solutions.drawing_utils


# cap = cv2.VideoCapture(0)
# with mp_iris.Iris() as iris:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = iris.process(image)

#     # Draw the eye and iris annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.face_landmarks_with_iris:
#         mp_draw.draw_iris_landmarks(image,results.face_landmarks_with_iris)
#     cv2.imshow('MediaPipe Iris', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])#compute mass center, pupil location
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 1)
    except:
        pass

# detector = dlib.get_frontal_face_detector()

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]#orginal height and width 640*480
    # rects = detector(gray, 1)#detect face in the 
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                        1.0, (300, 300), (104.0, 117.0, 123.0))

    detector.setInput(blob)
    faces = detector.forward()         
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]#same as faces[0][0][i][2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x,y,w,h) = box.astype("int")
            cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
            rect=dlib.rectangle(left=x, top=y, right=w, bottom=h)                         
    # for rect in rects:
    #     x=rect.left()
    #     y=rect.top()
    #     w=rect.right()-x
    #     h=rect.bottom()-y
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    

        shape = predictor(gray, rect)
        # shape = shape_to_np(shape)
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # mask = eye_on_mask(mask, left)
        # mask = eye_on_mask(mask, right)
        # mask = cv2.dilate(mask, kernel, 5)
        # eyes = cv2.bitwise_and(img, img, mask=mask)
        # mask = (eyes == [0, 0, 0]).all(axis=2)
        # eyes[mask] = [255, 255, 255]
        # mid = (shape[39][0] + shape[42][0]) // 2
        # eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        # threshold = cv2.getTrackbarPos('threshold', 'image')
        # _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        # thresh = cv2.erode(thresh, None, iterations=2) #1
        # thresh = cv2.dilate(thresh, None, iterations=4) #2
        # thresh = cv2.medianBlur(thresh, 3) #3
        # thresh=cv2.GaussianBlur(thresh,(3,3),0)
        # thresh = cv2.bitwise_not(thresh)
        # contouring(thresh[:, 0:mid], mid, img)
        # contouring(thresh[:, mid:], mid, img, True)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img)
    # cv2.imshow('eyes_gray',eyes_gray)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
