#!/usr/bin/env python
# coding=utf-8 
import cv2
import os    
import numpy as np
import glob 
import time
import math

#src_path = "/home/zhiyong/"
dir = "/home/apt/BackAngle/20190411_png/"
files=os.listdir(dir)
files.sort(key=lambda x:str(x[:-4]))
cv2.namedWindow("test")
classifier=cv2.CascadeClassifier("/home/apt/anan/cascade04172/cascade0417_24*24_20_15000.xml") 
#font = ImageFont.truetype('simsun.ttc',40)
font=cv2.FONT_HERSHEY_SIMPLEX

for file in files:
  if  file.endswith('png'):
  #读一张图片
    #i=111
    frame = cv2.imread(dir+str(file))
    #frame=cv2.flip(frame,-1)
    if frame is None:
        print (dir+str(file)+ "is not find")
    #print ("current frame is : image "+str(file))
    height, width=frame.shape[:2]
    size= (int(width*0.25), int(height*0.25))
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # frame1 = cv2.resize(image,size,cv2.INTER_AREA)
    # frame1=cv2.equalizeHist(frame1)

    #image=np.zeros(size,dtype=np.float16)
    
    #cv2.equalizeHist(image,image)
    # divisor=8
    # h,w=size
    # minSize=(w/divisor,h/divisor)
    carRects=classifier.detectMultiScale(image,1.1,4)#,0,Size(20, 20))
    if len(carRects)>0 :
      #count+=1
      for carRect in carRects:
        #minID=0
        x,y,w1,h1=carRect
        x=x
        y=y
        # w1=w1+40
        # h1=h1+40
        # roi=np.zeros( (h1,w1))
        # roi = frame[y:int(y+h1),int(x):int(x+w1)]   
        #i+=1
        #每一张图片中的每一个目标
        #cv2.imwrite("/media/apt/新加卷/张凯凯/zhangkaikai/new_Neg_Sam/181213/"+str(i)+str(file),roi)        
        # cv2.rectangle(frame,(x+20,y+20),((x+w1-20),(y+h1-20)),(0,0,255))
        cv2.rectangle(frame,(x,y),((x+w1),(y+h1)),(100,0,0),1)
    # else:_
    cv2.imwrite("/media/apt/DATA1/BSD/detect0418/"+"a"+str(file),frame)
    cv2.imshow("test",frame) 
    key=cv2.waitKey(30)
    c=chr(key&255)
    if c in ['q','Q',chr(80)]:
      break
cv2.destroyWindow("test")  








