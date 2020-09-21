import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('pedestrian3.avi')

out = cv2.VideoWriter(
	'output.avi',
	cv2.VideoWriter_fourcc(*'MJPG'),
	15.,
	(640,480))

while 1:
	ret,frame = cap.read()
	#we will use rezing and converting into gray for faster detection

	frame =  cv2.resize(frame,(640,480))
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	
	boxes, weights = hog.detectMultiScale(frame, winStride = (8,8))
	
	boxes = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
	
	for (xA,yA,xB,yB) in boxes:
		cv2.rectangle(frame,(xA,yA),(xB,yB),(0,255,0),2)
	
	out.write(frame)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('e'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey()

