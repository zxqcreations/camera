import cv2
import numpy as np
import glob
import time
import os
from xml.dom import minidom

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

w, h = 9, 6

objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

obj_points = []
img_points = []
ind_points = 0
cnt_images = 0

is_inf = 1

images = glob.glob("path for images\*.jpg")

cap = cv2.VideoCapture(1)

#for fname in images:
    #img = cv2.imread(fname)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);

conv_color = np.zeros([int(height), int(width), 3], np.uint8)
conv_color.fill(255)
gray = np.array((0))

while(True):
    if not cap.isOpened():
        print('failed to open cap')
        break
    ret, img = cap.read()

    if is_inf == 1:
        img = conv_color - img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    #ret, corners = cv2.findChessboardCorners(img, (w, h), None)

    cv2.putText(img, str(ind_points) + " set of points added," + \
                str(cnt_images) + "images saved", (5, 20), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    key = cv2.waitKey(20)

    if ret == True:
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        if key == 115: 
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            #cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            ind_points += 1
            print("New set of points added")

    cv2.imshow('corners', img)

    if key == 114:
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', \
                                    time.localtime(time.time()))
        cv2.imwrite(os.getcwd() + "\\cam_carlibration_img\\img_" + cur_time + "img.jpg", \
                    img)
        cv2.imwrite(os.getcwd() + "\\cam_carlibration_img\\img_" + cur_time + "gray.jpg", \
                    gray)
        cnt_images += 1

    if key == 32:
        break

#cap.release()
cv2.destroyAllWindows()

print("now start calibration")


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, \
                                                   gray.shape[::-1], None, None)
print('{:6s}'.format('mtx: \n'), mtx)
print('{:6s}'.format('dist: '), dist)

impl = minidom.getDOMImplementation()
dom = impl.createDocument(None, 'calibration', None)
root = dom.documentElement
cali = dom.createElement('matrix')
cali.setAttribute('id', str(is_inf))
root.appendChild(cali)

mat = dom.createElement('mat')
mat_data = dom.createTextNode(str(mtx[0,0])+','+
                              str(mtx[0,2])+','+
                              str(mtx[1,1])+','+
                              str(mtx[1,2]))
mat.appendChild(mat_data)
cali.appendChild(mat)

dis = dom.createElement('dist')
dis_data = dom.createTextNode(str(dist[0,0])+','+
                              str(dist[0,1])+','+
                              str(dist[0,2])+','+
                              str(dist[0,3])+','+
                              str(dist[0,4]))
dis.appendChild(dis_data)
cali.appendChild(dis)

if not is_inf:
    fn = os.getcwd() + '\\calibration.xml'
else:
    fn = os.getcwd() + '\\calibration_inf.xml'

with open(fn, 'a') as f:
    dom.writexml(f, addindent=' ', newl='\n')



print("Calibration finished")

#cap1 = cv2.VideoCapture(1)

while(True):
    if not cap.isOpened():
        print("failed to open cap for stage 2")
        break
    ret_cap, img_cap = cap.read()
    #print(img_cap.shape)
    img_h, img_w = img_cap.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, \
                                                      (img_w, img_h), 0,
                                                      (img_w, img_h))
    #print(newcameramtx)
    cv2.imshow('before_calibration', img_cap)
    dst = cv2.undistort(img_cap, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst1 = dst[y:y+h, x:x+w]
    cv2.imshow('calibration', dst)

    if cv2.waitKey(20) > 0:
        break
    
cv2.destroyAllWindows()
    
