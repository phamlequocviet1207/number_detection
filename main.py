import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

################
path = 'myData'
sizeImg = (32, 32)
testRatio = 0.2
valRatio = 0.2
################

images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)

print("Importing class .........")

count = 0
for x in range(noOfClasses):
    myPiclist = os.listdir(path + "/" + str(x))
    for y in myPiclist:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, sizeImg)
        images.append(curImg)
        classNo.append(x)
    print(count, end=" ")
    count += 1

print()
print("Total images in Images List = ", len(images))
print("Total IDS in classNo List = ", len(classNo))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for i in range(noOfClasses):       
    numOfSamples.append(len(np.where(y_train == i)[0]))
print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

print(X_train[30].shape)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# ImageDataGenerator for data augmentation
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

dataGen.fit(X_train)

# Convert labels to categorical
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))  # Fixed activation
    model.compile(optimizer=Adam(learning_rate=0.001),  # Fixed optimizer syntax
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and compile the model
model = myModel()
print(model.summary())