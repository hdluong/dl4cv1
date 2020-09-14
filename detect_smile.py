from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required = True,
	help = "path to where the face cascade resides")
ap.add_argument("-m", "--model", required = True,
	help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not supplied, use the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

while True:
	# Capture frame-by-frame
	ret, frame = camera.read()

	if args.get("video") and not ret:
		break

	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()

	rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	for (x,y,w,h) in rects:
		roi = gray[y:y+h, x:x+w]
		roi = cv2.resize(roi, (28,28))
		roi = roi.astype("float")/255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis = 0)

		(notSmiling, smiling) = model.predict(roi)[0]
		label = "Smiling" if smiling > notSmiling else "Not Smiling"

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frameClone, label, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (x, y), (x + w, y + h),
			(0, 0, 255), 2)

	# show our detected faces along with smiling/not smiling labels
	cv2.imshow("Face", frameClone)

	# if the ’q’ key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()