import numpy as np
import cv2
import os

################
path = 'myData'
sizeImg = (32,32)
################

images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)

print("Importing class .........")


for x in range (noOfClasses):
    myPiclist = os.listdir(path+"/"+str(x))
    for y in myPiclist:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, sizeImg)
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
 
print()

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)
    
