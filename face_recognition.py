import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from datetime import date , datetime




data=np.load("C:/Users/Dell/OneDrive/Desktop/vs/python/face_data.npy")

# print(data.shape)                                                

x=data[:,1:].astype(int)  
y=data[:,0]

model=KNeighborsClassifier()
model.fit(x,y)

import cv2 as cv
import numpy as np
import os

cap=cv.VideoCapture(0)
detector=cv.CascadeClassifier("C:/Users/Dell/OneDrive/Desktop/vs/python/haarcascade_frontalface_alt.xml")




while True:
    ret,frame=cap.read() 
        
    if ret:
        faces=detector.detectMultiScale(frame)
        
        for face in faces:
            x,y,w,h=face
            cut=frame[y:y+h,x:x+w]
            fix=cv.resize(cut,(100,100))
            gray=cv.cvtColor(fix,cv.COLOR_BGR2GRAY)
            out=model.predict([gray.flatten()])
            # print(out)
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
            cv.putText(frame,str(out[0]),(x,y-10),cv.FONT_HERSHEY_SIMPLEX,2,(255,0,0),4)
            cv.imshow("face", gray)
            
            
           
 
    cv.imshow("window", frame)
        
    
    key=cv.waitKey(1)     
    
    if key == ord("q"):
        break



    
cap.release()
cv.destroyAllWindows()
now = datetime.now()

df = pd.DataFrame({'Name': out,'Date':date.today() ,'Time': now.strftime("%H:%M:%S")
})
df.to_csv("C:/Users/Dell/Downloads/attendance monitor.csv", mode='a', index=False, header=not os.path.isfile("C:/Users/Dell/Downloads/attendance monitor.csv"))   


print(out)