import cv2
import time

first_frame=None
face=cv2.CascadeClassifier("E:\PROJECTS\FaceDetection\Extra_files\haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(cv2.CAP_DSHOW)
while True:
    _,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if first_frame is None:
        first_frame=gray
        continue
    faces=face.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key==ord("e"):
        break



video.release()
cv2.destroyAllWindows()