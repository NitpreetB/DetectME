import cv2 as cv
import mediapipe as np 
import time 

class poseDetector():
    def __init__(self,mode=False, upBody=False,smooth=True,detectionCon=False,trackCon=0.5):
        
        self.mode=mode
        self.upBody=upBody
        self.smooth= smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpDraw = np.solutions.drawing_utils
        self.mpPose=np.solutions.pose
        self.pose= self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)

    def findPose(self,img,draw=True):

        imgRGB= cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self,img, draw=True):
        lmList=[]
        if self.results.pose_landmarks: 
            for id, lm in enumerate( self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print (id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,0),cv.FILLED)
        return lmList

def main():
    cap = cv.VideoCapture('Videos/Zion Williamson dominates high school dunk contest _ Powerade Jam Fest.mp4')
    pTime=0
    detector=poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList=detector.getPosition(img)
        print(lmList)

        cTime= time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv.putText(img,str(int(fps),),(70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),thickness=3)

        cv.imshow('image',img)
        cv.waitKey(1)


if __name__== "__main__":
    main()