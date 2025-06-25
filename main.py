import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split


################
path = 'myData'
sizeImg = (32,32)
testRatio = 0.2
valRatio = 0.2
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
    

# Splitting the data
X_train,X_test,y_train, y_test = train_test_split(images, classNo,test_size=testRatio)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=valRatio)


print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
