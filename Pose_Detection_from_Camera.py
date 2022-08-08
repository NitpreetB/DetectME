import time

import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
if not (cap.isOpened()):
    print ('camera could not be opened')

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
pTime=0

mpDraw = mp.solutions.drawing_utils
mpPose=mp.solutions.pose 
pose= mpPose.Pose()

while True:
    success, img = cap.read()
    imgRGB= cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    ##print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate( results.pose_landmarks.landmark):
            h,w,c = img.shape
            print (id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv.circle(img,(cx,cy),5,(255,0,0),cv.FILLED)


   
    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv.putText(img,str(int(fps),),(70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),thickness=3)

    cv.imshow('image',img)
    cv.waitKey(1)


