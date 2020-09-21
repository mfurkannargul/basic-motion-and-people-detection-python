import cv2
import numpy as np

def motion_detection():

	cap = cv2.VideoCapture(0)

	#resolution = (500,500)

	if cap.isOpened():

		ret,frame = cap.read()

	else:
		ret = False

	ret,frame1 = cap.read()
	ret,frame2 = cap.read()

	while ret:
		ret,frame = cap.read()

		difference = cv2.absdiff(frame1,frame2)

		grey_scale = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(grey_scale,(5,5),0)
		ret,threshold = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
		kernel = np.ones((5,5),np.uint8)
		dilate = cv2.dilate(threshold, kernel, iterations = 3)
		image, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for c in contours:
			if cv2.contourArea(c) <= 500:
				continue

			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(frame1, (x,y), (x+w, y+h), (255,0,0), 2)		
			v = int(w/2)

			cv2.circle(frame1,(x+v,y+h), 5, (0,255,0), -2)

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame1, 'Status: DETECTED', (10,50), font, 1, (0,0,255), 2, cv2.LINE_AA)

		#cv2.drawContours(frame1,contours,-1,(255,0,0),2)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame1, 'Status:', (10,50), font, 1, (0,0,255), 2, cv2.LINE_AA)

		cv2.imshow("frame2",frame2)
		cv2.imshow("frame1",frame1)

		if cv2.waitKey(40) == 27:
			break
		frame1 = frame2
		ret,frame2 = cap.read()
	cv2.destroyAllWindows()
	#VideoFileOutput.release()
	cap.release()

motion_detection()
