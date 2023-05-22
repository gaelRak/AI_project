'''import cv2
image_path="image/created/drawn_image.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=6)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = image[y:y+h, x:x+w]
    image_path_out="image/created/ROI.png"
    cv2.imwrite(image_path_out, ROI)
    break

cv2.imshow('ROI', ROI)
cv2.waitKey()'''
import numpy as np
import cv2

# Load image and HSV color threshold
image_path="image/test/a01-000x-s00-00.png"
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([90, 38, 0])
upper = np.array([145, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(image, image, mask=mask)
result[mask==0] = (255, 255, 255)

# Find contours on extracted mask, combine boxes, and extract ROI
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = np.concatenate(cnts)
x,y,w,h = cv2.boundingRect(cnts)
cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
ROI = result[y:y+h, x:x+w]

cv2.imshow('result', result)
cv2.imshow('mask', mask)
cv2.imshow('image', image)
cv2.imshow('ROI', ROI)
cv2.waitKey()