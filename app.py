import time

import cv2 as cv
import mediapipe as mp
import glob
import os
import sys
from subprocess import call
from moviepy.editor import VideoFileClip
from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash, jsonify, abort
from werkzeug.utils import secure_filename
from PIL import Image
import PIL.Image
import yolov5.detect as dt
from src.config import shooting_result
from src.app_helper import getVideoStream, get_image, detectionAPI
from tkinter import * 
from tkinter import filedialog



app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#useless key, in order to use session
app.secret_key = "super secret key" 

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/sampleobjectdetection', methods=['GET', 'POST'])
def upload_sample_image():

    os.system( "python yolov5/detect.py --source=0") 

    videoClip = VideoFileClip("../Human pose tracking/yolov5/runs/detect/exp/0.mp4")
    videoClip.write_gif("../Human pose tracking/static/img/gif2.gif")
    w1 = VideoFileClip("../Human pose tracking/yolov5/runs/detect/exp/0.mp4").w
    h1 =VideoFileClip("../Human pose tracking/yolov5/runs/detect/exp/0.mp4").h
    print (w1,"x", h1)
    return render_template("objectdetection.html")


@app.route('/sample_poseanalysis', methods=['GET', 'POST'])
def Live_video():

    files = glob.glob('../Human pose tracking/static/img/poseGIF/*')
    for f in files:
        os.remove(f)

    import time

    import cv2 as cv
    from cv2 import VideoWriter
    import mediapipe as mp

    cap = cv.VideoCapture(0)
    if not (cap.isOpened()):
        print ('camera could not be opened')

    fps = int(cap.get(cv.CAP_PROP_FPS)) #video source fps 
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
    #video source pixel width 
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) #video source pixel height
    results2 = VideoWriter('demo2.mp4',cv.VideoWriter_fourcc(*'MP4V'),fps,(width,height),True)
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
                cx,cy=int(lm.x*w),int(lm.y*h)
                cv.circle(img,(cx,cy),5,(255,0,0),cv.FILLED)


    
        cTime= time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        ##cv.putText(img,str(int(fps),),(70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),thickness=3)
        cv.putText(img,'Press x to Exit',(70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),thickness=3)

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

    video_capture = cv.VideoCapture('../Human pose tracking/demo2.mp4')
    still_reading, image = video_capture.read()
    frame_count = 0
    while still_reading:
        cv.imwrite(f"../Human pose tracking/static/img/poseGIF/frame_{frame_count:03d}.jpg", image)
        
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1
    
    images = glob.glob(f"../Human pose tracking/static/img/poseGIF/*.jpg")
    images.sort()
    frames = [PIL.Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save("../Human pose tracking/static/img/gif1.gif", format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)
    
    print("Gif has been saved") 

    files = glob.glob('../Human pose tracking/static/img/poseGIF/*')
    for f in files:
        os.remove(f)

    return render_template("posedetection.html")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)