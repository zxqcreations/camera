import cv2

cap = cv2.VideoCapture(1)

while(True):
    if not cap.isOpened():
        print('failed to open cap')
        break
    ret, img = cap.read()

    cv2.imshow('t1',img)
    if cv2.waitKey(20) > 0:
        break

cv2.destroyAllWindows()

while(True):
    if not cap.isOpened():
        print('failed to open cap')
        break
    ret1, img1 = cap.read()

    cv2.imshow('t2',img1)
    if cv2.waitKey(20) > 0:
        break

cv2.destroyAllWindows()
