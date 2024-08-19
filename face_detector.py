# importing open cv libraray
import cv2

# dataset load
trainedData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# start of webcam
cap =cv2.VideoCapture(0)

while True:  # to get multiple frame
    success , frame =cap.read()

    # conversion to black and white(grayscale)
    greyImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detect faces
    faceCoordinates = trainedData.detectMultiScale(greyImg)  # to get the coordinates

    for (x , y , w , h) in faceCoordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('window',frame)
    key = cv2.waitKey(1)         # frame changed in every 1 milqisec(0-for pic & 1-for video)

    # enter q to stop the webcam
    if(key==81 or key==113):
        break

cap.release()
cv2.destroyAllWindows()
print("End of program")

