import cv2
import numpy as np

filename = 'C:\\workspace\\video source\\out4.avi'
filename_inf = 'C:\\workspace\\video source\\out_inf4.avi'
cap = cv2.VideoCapture(filename)
cap_inf = cv2.VideoCapture(filename_inf)

while(1):
    if not cap.isOpened():
        print('cannot open video file')
        break
    if not cap_inf.isOpened():
        print('cannot open inf video file')
        break
    ret, img = cap.read()
    ret_inf, img_inf = cap_inf.read()
    if not ret:
        break
    if not ret_inf:
        break
    cv2.imshow('test', img)
    cv2.imshow('test_inf', img_inf)

    if cv2.waitKey(20) > 0:
        break

cv2.destroyAllWindows()
