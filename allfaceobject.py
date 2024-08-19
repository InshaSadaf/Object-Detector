import numpy as np
import cv2
from deepface import DeepFace

prototxt_path ='MobileNetSSD_deploy.prototxt.txt'
model_path = 'MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2


classes = ["background", "aeroplane", "bicycle", "bird", " boat"
           , "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog",
           "horse", "motorbike", "person", " potted plant ", "sheep", "sofa", "train", " tvmoniter ", "cellphone" ,
           "cup","fork", "knife", "spoon", "bowl", "mouse",
            " remote", "book"]

np.random.seed(543210)
colors = np.random.uniform(0, 255,  size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()

    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):

        confidence = detected_objects[0][0][i][2]

        if confidence > min_confidence:

            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_left_x = int(detected_objects[0, 0, i, 5] * width)
            lower_left_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_left_x, lower_left_y), colors[class_index], 3)
            cv2.putText(image, prediction_text,
                        (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    cv2.imshow("Detected Objects", image)
    key = cv2.waitKey(1)
    if (key == 81 or key == 113):
        break

cv2.destroyAllWindows()
cap.release()
