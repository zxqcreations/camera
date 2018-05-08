import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image

cap = cv2.VideoCapture(1)
cap_inf = cv2.VideoCapture(0)

while(True):
    if not cap.isOpened():
        print('Failed to open stream')
        break
    if not cap_inf.isOpened():
        print('Failed to open stream')
        break
    ret, img = cap.read()
    ret_inf, img_inf = cap_inf.read()

    #-------------------------------- Image Enhance Operation
    img_inf = Image.fromarray(img_inf)

    #enb = ImageEnhance.Contrast(img_inf)
    #img_inf = enb.enhance(1.5)
    #enb = ImageEnhance.Color(img_inf)
    #img_inf = enb.enhance(4)
    #enb = ImageEnhance.Sharpness(img_inf)
    #img_inf = enb.enhance(8)

    img_inf = np.asarray(img_inf)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inf = cv2.cvtColor(img_inf, cv2.COLOR_BGR2GRAY)
    #img[:,:,0] = 0
    #img[:,:,1] = 0
    #-------------------------------- Image Enhance Operation
    #img_inf = cv2.GaussianBlur(img_inf, (3, 3), 0)
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))

    sobel_X = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_X = np.uint8(np.absolute(sobel_X))
    
    sobel_Y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel_Y = np.uint8(np.absolute(sobel_Y))

    sobel_XY = cv2.bitwise_or(sobel_X, sobel_Y)

    sobel_X_inf = cv2.Sobel(gray_inf, cv2.CV_64F, 1, 0)
    sobel_X_inf = np.uint8(np.absolute(sobel_X_inf))
        
    sobel_Y_inf = cv2.Sobel(gray_inf, cv2.CV_64F, 0, 1)
    sobel_Y_inf = np.uint8(np.absolute(sobel_Y_inf))
    
    sobel_XY_inf = cv2.bitwise_or(sobel_X_inf, sobel_Y_inf)

    canny = cv2.Canny(gray, 50, 150)
    canny_inf = cv2.Canny(gray_inf, 50, 150)

    #-------------------------------- Pattern Recognization
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(sobel_XY, None)

    #kp, des = sift.detectAndCompute(gray, None)
    
    #sobel_XY = cv2.drawKeypoints(sobel_XY, kp, sobel_XY)

    kp_inf, des_inf = sift.detectAndCompute(sobel_XY_inf, None)

    #kp_inf, des_inf = sift.detectAndCompute(gray_inf, None)

    #sobel_XY_inf = cv2.drawKeypoints(sobel_XY_inf, kp_inf, sobel_XY_inf)

    bf = cv2.BFMatcher_create(cv2.NORM_L2)
    matches = bf.knnMatch(des, des_inf, k = 2)
    goodmatch = []
    out = np.array((1,1,1))
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodmatch.append((m, n))
    #print(matches)
    #print(goodmatch)
    
    out = cv2.drawMatchesKnn(sobel_XY, kp, sobel_XY_inf,
                             kp_inf, goodmatch[:5], None, flags = 2)

    #out = cv2.drawMatchesKnn(gray, kp, gray_inf,
    #                         kp_inf, goodmatch[:5], None, flags = 2)
    
    cv2.imshow('Original', img)
    #cv2.imshow('Gray', gray)
    cv2.imshow('Gray_inf', gray_inf)
    #cv2.imshow('Laplacian', lap)
    #cv2.imshow('Sobel X', sobel_X)
    #cv2.imshow('Sobel Y', sobel_Y)
    cv2.imshow('Sobel XY', sobel_XY)
    #cv2.imshow('Canny', canny)
    cv2.imshow('Sobel_inf XY', sobel_XY_inf)
    #cv2.imshow('Canny_inf', canny_inf)
    #cv2.imshow('Outer', out)
    
    

    if cv2.waitKey(20) > 0:
        break

cv2.destroyAllWindows()
