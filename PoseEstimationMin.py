import time

import cv2 as cv
from cv2 import VideoWriter
import mediapipe as mp

cap = cv.VideoCapture('../yolov5/IMG_2.mp4')
fps = int(cap.get(cv.CAP_PROP_FPS)) #video source fps 
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
#video source pixel width 
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) #video source pixel height
results2 = VideoWriter('demo1.mp4',cv.VideoWriter_fourcc(*'MP4V'),fps,(width,height),True)
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


    #scale_percent = 60 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    
    # resize image
   # resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    results2.write(img)
    cv.imshow('image',img)
    if cv.waitKey(1) & 0xFF == ord('x'):
      break


# release video capture
# and video write objects
cap.release()
results2.release()

# Closes all the frames
cv.destroyAllWindows() 

print("The video was successfully saved") 