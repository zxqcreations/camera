import numpy as np
import cv2

img = np.zeros((630,891,3), np.uint8)
img.fill(0)

for i in range(0, 629):
    for j in range(0, 890):
        if (i // 90 + j // 90) % 2 == 0:
            img[i, j].fill(255)

cv2.imshow('image',img)

cv2.imwrite('chess.jpg',img)

if cv2.waitKey(20) > 0:
    cv2.destroyAllWindows()
