import cv2
print("insha")
img=cv2.imread("pic.png", 1)#("filename",0/1)->0=b&w 1->colourful
cv2.imshow("output",img)
cv2.waitKey(0)#infinite delay
cv2.destroyAllWindows()
