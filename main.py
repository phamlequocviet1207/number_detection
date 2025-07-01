import numpy as np
import cv2
import pickle

############
width = 640
height = 480
imgSize = (32,32)
############

cap = cv2.VideoCapture(1)
cap.set(3,width)
cap.set(4,height)

pickle_in = open("output/model_train.p","rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    ret, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img,imgSize)
    img = preProcessing(img)
    cv2.imshow("Processed Image",img)
    img = img.reshape(1,32,32,1)

    #Predict 
    classIndex = int(model.predict_classes(ing))
    print(classIndex)
    prediction = model.predict(img)
    print(prediction)
    probVal = np.amax(prediction)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

