import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

count = 0
for x in range (noOfClasses):
    myPiclist = os.listdir(path+"/"+str(x))
    for y in myPiclist:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, sizeImg)
        images.append(curImg)
        classNo.append(x)
    print(count,end=" ")
    count += 1
 
print()
print("Total images in Images List = ", len(images))
print("Total IDS in classNo List = ", len(classNo))

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

numOfSamples = []
for i in range(noOfClasses):       
    # print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

print(X_train[30].shape)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img



X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

# img = X_train[30] 
# img = cv2.resize(img, (300,300))
# cv2.imshow("Preprocessing",img)
# cv2.waitKey(0) 

# print(X_train[30].shape)

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2],1)
