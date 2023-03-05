import cv2
import pickle
import numpy as np

# Video feed
cap = cv2.VideoCapture("carPark.mp4")

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)
    print(posList)

width, height = 30, 65


def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)
        
        if count < 320:
            color = (0, 255, 0)
            thickness = 2
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cv2.putTextRect(img, str(count), (x, y + height - 1), scale=1,
                           thickness=1, offset=1, colorR=color)

    cv2.putTextRect(img, f'Empty: {spaceCounter}/{len(posList)}', (20, 40), scale=2,
                           thickness=2, offset=6, colorR=(200,120,0))
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    cv2.imshow("ImgMedian", imgMedian)
    cv2.imshow("ImgDilate", imgDilate)
    cv2.imshow("ImgBlur", imgBlur)
    cv2.imshow("ImgThres", imgMedian)
    cv2.waitKey(15)