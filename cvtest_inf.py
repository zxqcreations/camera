import cv2
import time
import os
import numpy as np
from PIL import ImageEnhance
from PIL import Image


#cap = cv2.VideoCapture(2)
cap_inf = cv2.VideoCapture(0)

rect_x1, rect_y1, rect_x2, rect_y2, current_x, current_y = -1, -1, -1, -1, -1, -1

def getCurrentTime():
    return time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime(time.time()))

def mouse_event(event, x, y, flags, param):
    global cap, cap_inf, rect_x1, rect_y1, rect_x2, \
           rect_y2, current_x, current_y
    if event == cv2.EVENT_MOUSEMOVE:
        current_x, current_y = x, y

#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
width_inf = cap_inf.get(cv2.CAP_PROP_FRAME_WIDTH);
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
height_inf = cap_inf.get(cv2.CAP_PROP_FRAME_HEIGHT);

is_conv = 0

ret_inf, inf_frame = cap_inf.read()
#print(inf_frame)
conv_color = np.zeros([512, 640, 3], np.uint8)
conv_color.fill(255)
#print(conv_color)

while(True):
    #if not cap.isOpened():
    #    print('failed to open stream')
    #    break
    if not cap_inf.isOpened():
        print('failed to open inf stream')
    
    #ret, frame = cap.read()
    ret_inf, inf_frame = cap_inf.read()

    inf_tmp = inf_frame
    
    #inf_cvt = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2RGB)
    #print(type(inf_frame))
    #print(np.asarray(inf_frame).shape)
    inf_frame = Image.fromarray(inf_frame)
    enb = ImageEnhance.Contrast(inf_frame)
    inf_frame = enb.enhance(1)
    
    enb = ImageEnhance.Color(inf_frame)
    inf_frame = enb.enhance(4)

    enb = ImageEnhance.Sharpness(inf_frame)
    inf_frame = enb.enhance(8)

    #inf_frame = cv2.cvtColor(np.asarray(inf_frame),cv2.COLOR_RGB2BGR)

    inf_frame = np.asarray(inf_frame)

    if is_conv == 1:
        inf_frame = conv_color - inf_frame

    #cv2.putText(frame, str(width) + ' ' + str(height) + ' ' + str(current_x) + ' ' +  \
    #            str(current_y), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    cv2.putText(inf_frame, str(width_inf) + ' ' + str(height_inf) + ' ' + str(current_x) + ' ' + \
                str(current_y), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    #The third arg is for position (x, y) and the last arg is for color, which formatted as BGR

    #cv2.imshow('video', frame)
    cv2.setMouseCallback('video', mouse_event)
    cv2.imshow('video_inf', inf_frame)
    cv2.imshow('video_inf_original', inf_tmp)

    cv_chr = cv2.waitKey(5)
    #print(str(cv_chr))

    if cv_chr == 32:
        break
    if cv_chr == 115:
    #    print(os.getcwd())
        cv2.imwrite(os.getcwd() + "\\cv_img\\img_" + \
                    getCurrentTime() + ".jpg", inf_frame)
    #if cv2.waitKey(115) >0:
    #    print("s is pressed")
    #if cv2.waitKey(20) > 0:
        #break

cap_inf.release()
cv2.destroyAllWindows()
