import cv2
import os
import numpy as np
from xml.dom import minidom


cap = cv2.VideoCapture(1)
cap_inf = cv2.VideoCapture(0)

rect_x1, rect_y1, rect_x2, rect_y2, current_x, current_y = -1, -1, -1, -1, -1, -1


def mouse_event(event, x, y, flags, param):
    global cap, cap_inf, rect_x1, rect_y1, rect_x2, rect_y2, current_x, current_y
    if event == cv2.EVENT_MOUSEMOVE:
        current_x, current_y = x, y

mat_vis = np.zeros((3,3))
dist_vis = np.zeros((1,5))
mat_inf = np.zeros((3,3))
dist_inf = np.zeros((1,5))

pause = False

with open(os.getcwd() + '\\calibration.xml', 'r') as f:
    dom = minidom.parse(f)
    root = dom.documentElement
    mat = root.getElementsByTagName('matrix')[0]
    data = mat.getElementsByTagName('mat')[0].childNodes[0].data
    dist_data = mat.getElementsByTagName('dist')[0].childNodes[0].data
    
    mat_vis[0,0] = float(data.split(',')[0])
    mat_vis[0,2] = float(data.split(',')[1])
    mat_vis[1,1] = float(data.split(',')[2])
    mat_vis[1,2] = float(data.split(',')[3])
    mat_vis[2,2] = 1
    print(mat_vis)
    dist_vis[0,0] = float(dist_data.split(',')[0])
    dist_vis[0,1] = float(dist_data.split(',')[1])
    dist_vis[0,2] = float(dist_data.split(',')[2])
    dist_vis[0,3] = float(dist_data.split(',')[3])
    dist_vis[0,4] = float(dist_data.split(',')[4])
    print(dist_vis)

with open(os.getcwd() + '\\calibration_inf.xml', 'r') as f:
    dom = minidom.parse(f)
    root = dom.documentElement
    mat = root.getElementsByTagName('matrix')[0]
    data = mat.getElementsByTagName('mat')[0].childNodes[0].data
    dist_data = mat.getElementsByTagName('dist')[0].childNodes[0].data
    
    mat_inf[0,0] = float(data.split(',')[0])
    mat_inf[0,2] = float(data.split(',')[1])
    mat_inf[1,1] = float(data.split(',')[2])
    mat_inf[1,2] = float(data.split(',')[3])
    mat_inf[2,2] = 1
    print(mat_inf)
    dist_inf[0,0] = float(dist_data.split(',')[0])
    dist_inf[0,1] = float(dist_data.split(',')[1])
    dist_inf[0,2] = float(dist_data.split(',')[2])
    dist_inf[0,3] = float(dist_data.split(',')[3])
    dist_inf[0,4] = float(dist_data.split(',')[4])
    print(dist_inf)
    

while(True):
    if not cap.isOpened():
        print('failed to open stream')
        break
    if not cap_inf.isOpened():
        print('failed to open inf stream')
        break

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    width_inf = cap_inf.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    height_inf = cap_inf.get(cv2.CAP_PROP_FRAME_HEIGHT);

    if not pause:
        ret, frame = cap.read()
        ret_inf, inf_frame = cap_inf.read()

    img_h, img_w = frame.shape[:2]
    inf_h, inf_w = inf_frame.shape[:2]

    newImgMatrix, img_roi = cv2.getOptimalNewCameraMatrix(mat_vis, dist_vis, \
                                                          (img_w, img_h), 0, \
                                                          (img_w, img_h))
    newInfMatrix, inf_roi = cv2.getOptimalNewCameraMatrix(mat_inf, dist_inf, \
                                                          (inf_w, inf_h), 0, \
                                                          (inf_w, inf_h))
    dst_vis = cv2.undistort(frame, mat_vis, dist_vis, None, newImgMatrix)
    x, y, w, h = img_roi
    dst_vis1 = dst_vis[y:y+h, x:x+w]

    dst_inf = cv2.undistort(inf_frame, mat_inf, dist_inf, None, newInfMatrix)
    x, y, w, h = inf_roi
    dst_inf1 = dst_inf[y:y+h, x:x+w]

    test_img = dst_vis[80:370, 115:460, :]
    #print(inf_frame.shape[0], inf_frame.shape[1])
    #shape = (dst_inf1.shape[1], dst_inf1.shape[0])
    #test_img = cv2.resize(test_img, shape, \
    #                      interpolation=cv2.INTER_NEAREST)
    shape = (test_img.shape[1], test_img.shape[0])
    test_inf = cv2.resize(dst_inf1, shape, \
                          interpolation=cv2.INTER_NEAREST)

    cv2.putText(test_img, str(width) + ' ' + str(height) + ' ' + str(current_x) + ' ' +  \
                str(current_y), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    cv2.circle(test_inf,(current_x, current_y), 1,(0, 0, 255))
    #cv2.putText(dst_inf1, str(width_inf) + ' ' + str(height_inf) + ' ' + str(current_x) + ' ' + \
    #            str(current_y), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    #The third arg is for position (x, y) and the last arg is for color, which formatted as BGR

    #cv2.imshow('video', dst_vis1)
    #cv2.imshow('video_inf', dst_inf1)
    cv2.imshow('test_inf', test_inf)
    cv2.imshow('test_img', test_img)
    cv2.setMouseCallback('test_img', mouse_event)

    key = cv2.waitKey(20)
    if key == 32:
        break
    if key == 115:
        if not pause:
            pause = True
        else:
            pause = False

cap_inf.release()
cv2.destroyAllWindows()
