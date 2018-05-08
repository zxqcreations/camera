import cv2
import numpy as np

img = np.zeros([560,560,3], np.uint8)
img.fill(0)

for i in range(0, 559):
    for j in range(0, 559):
        if not ((i//100)+(j//100))%2 == 0:
            img[i,j]=[0,160,255]

for i in range(0, 28):
    cv2.line(img, (i * 20, 0), (i * 20, 559), (255, 138, 68))

for i in range(0, 28):
    cv2.line(img, (0, i * 20), (559, i * 20), (255, 0, 0))


cv2.imshow('img',img)
