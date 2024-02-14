import cv2 as cv
import numpy as np
import os

cap=cv.VideoCapture(0)
detector=cv.CascadeClassifier("C:/Users/Dell/OneDrive/Desktop/vs/python/haarcascade_frontalface_alt.xml")

name=input("Enter your name: ")


frames=[]
output=[]

while True:
    ret,frame=cap.read() 
        
        
    if ret:
        faces=detector.detectMultiScale(frame)
        
        for face in faces:
            x,y,w,h=face
            cut=frame[y:y+h,x:x+w]
            fix=cv.resize(cut,(100,100))
            gray=cv.cvtColor(fix,cv.COLOR_BGR2GRAY)
 
        cv.imshow("window", frame)
        # cv.imshow("face", gray)
        
        
    key=cv.waitKey(1)     
    
    if key == ord("q"):
        break
    
    if key==ord("c"):
        #cv.imwrite(name+".jpg",frame)
        frames.append(gray.flatten())
        output.append([name])
    

x=np.array(frames)
y=np.array(output)

data=np.hstack([y,x])


f_name="face_data.npy"

if os.path.exists("C:/Users/Dell/OneDrive/Desktop/vs/python/face_data.npy"):
    old=np.load("C:/Users/Dell/OneDrive/Desktop/vs/python/face_data.npy")
    data=np.vstack([old,data])

np.save("C:/Users/Dell/OneDrive/Desktop/vs/python/face_data.npy",data)

cap.release()
cv.destroyAllWindows()