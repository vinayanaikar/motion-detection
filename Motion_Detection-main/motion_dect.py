import cv2,time
import numpy as np
from datetime import datetime
import pandas as pd

first_frame = None
status_list = [None,None]
times =[]
df = pd.DataFrame(columns=["START","END"])

vedio = cv2.VideoCapture(0)

while True:
    check ,frame =vedio.read()
    status= 0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame =cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations=0)
    cnts,_ = cv2.findContours(thresh_delta.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) <10000:
            continue
        status=1
        (x,y,w,h) =cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    status_list = status_list[-2:]

    #record datetime when changes occur
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("frame",frame)
    # cv2.imshow("capturing", gray)
    # cv2.imshow("delta", delta_frame)
    # cv2.imshow("thres", thresh_delta)
    key =cv2.waitKey(1)
    if key == ord('q'):
        break
print(status_list)
print(times)

for i in range(0,len(times)-1,3):
    cv2.putText(frame, "Status:{}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    df = df.append({"START": times[i],"END":times[i+1],"DURATION":times[i+1]-times[i]},ignore_index=True)
df.to_csv("Times.csv")
vedio.release()
cv2.destroyAllWindows()

