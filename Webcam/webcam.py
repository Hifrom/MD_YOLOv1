import cv2
import numpy

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# WarmUp
for i in range(0, 65):
    ret, frame = cap.read()

i = 312
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (416, 416))
    cv2.imshow('frame',frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    print('Record Image: {0}.jpg'.format(str(i)))
    cv2.imwrite('images_webcam2/' + str(i) + '_test.jpg', frame)
    i += 1
cap.release()
cv2.destroyAllWindows()