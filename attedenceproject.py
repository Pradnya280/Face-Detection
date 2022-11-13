
from pydoc import classname
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path  = 'ImageBasic'
Images = []
classNames = []
mylist = os.listdir(path)
print(mylist)


for cl in mylist:
    curImg =cv2.imread(f'{path}/{cl}')#cl name of image
    Images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findencodings(Images):
    encodeList = []
    for img in Images:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendence.csv','r+') as f:
        myDataList =f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')





encodeListKnown = findencodings(Images)
print('Encoding complete')


cap = cv2.VideoCapture(0)
while True:
    success, img  = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    for encodeFace, faceloc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)








#faceloc = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
#cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

#facelocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

#we use linear svm to find results wheither they match or not
#results = face_recognition.compare_faces([encodeElon],encodeTest)
#faceDis = face_recognition.face_distance([encodeElon],encodeTest)
#print(results,faceDis)

#imgElon = face_recognition.load_image_file('ImageBasic/elon musk.jpg')
#imgElon  = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

#imgTest = face_recognition.load_image_file('ImageBasic/Bill gates.jpg')
#imgTest  = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)