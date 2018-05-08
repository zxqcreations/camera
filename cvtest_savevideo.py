import cv2


cap = cv2.VideoCapture(2)
cap_inf = cv2.VideoCapture(1)

if not cap.isOpened():
    print('failed to open stream')
    exit(1)
if not cap_inf.isOpened():
    print('failed to open inf stream')
    exit(1)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
width_inf = cap_inf.get(cv2.CAP_PROP_FRAME_WIDTH);
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
height_inf = cap_inf.get(cv2.CAP_PROP_FRAME_HEIGHT);

writer = cv2.VideoWriter('out.avi',-1, 20.0, (int(width), int(height)))
writer_inf = cv2.VideoWriter('out_inf.avi',-1, 20.0, (int(width_inf),
                                                      int(height_inf)))

rect_x1, rect_y1, rect_x2, rect_y2, current_x, current_y = -1, -1, -1, -1, -1, -1


def mouse_event(event, x, y, flags, param):
    global cap, cap_inf, rect_x1, rect_y1, rect_x2, rect_y2, current_x, current_y
    if event == cv2.EVENT_MOUSEMOVE:
        current_x, current_y = x, y
    

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
    
    ret, frame = cap.read()
    ret_inf, inf_frame = cap_inf.read()

    frame = cv2.flip(frame, 1)
    writer.write(frame)
    inf_frame = cv2.flip(inf_frame, 1)
    writer_inf.write(inf_frame)

    cv2.putText(frame, str(width) + ' ' + str(height) + ' ' + str(current_x) + ' ' +  \
                str(current_y), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    cv2.putText(inf_frame, str(width_inf) + ' ' + str(height_inf) + ' ' + str(current_x) + ' ' + \
                str(current_y), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    #The third arg is for position (x, y) and the last arg is for color, which formatted as BGR

    cv2.imshow('video', frame)
    cv2.setMouseCallback('video', mouse_event)
    cv2.imshow('video_inf', inf_frame)
    if cv2.waitKey(20) > 0:
        break

cap.release()
cap_inf.release()
writer_inf.release()
writer.release()
cv2.destroyAllWindows()
