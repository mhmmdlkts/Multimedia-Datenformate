import pillow_avif
from matplotlib.image import imread
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image

#img = imread('faces.avif')
img = imread('faces.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect facesRGB_img
faces = face_cascade.detectMultiScale(gray, 1.1, 6)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()

