import cv2
import sys

imagepath="E:\PROJECTS\FaceDetection\Extra_files\download (1).jpg"
#imagepath="E:\PROJECTS\FaceDetection\Extra_files\download.jpg"
img=cv2.imread(imagepath)
facecascade=cv2.CascadeClassifier("E:\PROJECTS\FaceDetection\Extra_files\haarcascade_frontalface_default.xml")

#resized=cv2.resize(img,(600,600))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=facecascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

print("Found {0} faces!".format(len(faces)))

for(x,y,w,h)in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Faces found",img)
cv2.waitKey(0)