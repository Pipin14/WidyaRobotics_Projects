import cv2
import numpy as np

#Dataset path
path = "data/cat.jpg"
img = cv2.imread(path)

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
            
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
            
        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = img[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
        #Image Color
        imgColor = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.cvtColor(imgColor, cv2.COLOR_GRAY2BGR)
        imgNegative = cv2.bitwise_not(roi, cv2.COLOR_BGR2BGR555)
        imgConcat = cv2.vconcat([roi, imgGray, imgNegative])
        cv2.imshow("Concatenate", imgConcat)
        cv2.imwrite('data/colorCat.png', roi) 
        cv2.imwrite('data/grayCat.png', imgGray)  
        cv2.imwrite('data/negativeCat.png', imgNegative) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #Resized image
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Resized image", resized)
        oriImg = img.copy()
        cv2.imwrite('data/resizeCat.png', resized)   
        cv2.waitKey(0)
        cv2.destroyAllWindows()      
          
cv2.namedWindow("cropped")
cv2.setMouseCallback("cropped", mouse_crop)

while True:

    i = img.copy()

    if not cropping:
        cv2.imshow("cropped", img)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("cropped", i)
    cv2.waitKey(0)
    cv2.imwrite('data/original.png', i)
    
# close all open windows
cv2.destroyAllWindows()