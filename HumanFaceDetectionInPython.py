# Import open cv
import cv2

# Loads the cascade classifier to be used for face detection.
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#This will capture the video from your default camera.
video_capture = cv2.VideoCapture(0)
img_counter = 0
while True:
    #Capture frame-by-frame
    ret, frame = video_capture.read()
    # Next convert to gray as cv2 processes most data in grayscale format.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    # Next we use the cascade classifier to detect the faces in the frame.
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # For all the detected faces we are going to draw rectangle around it.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Show the frame with rectangle now.
    cv2.imshow('FaceDetection', frame)
    #ESC Pressed: This will break the while loop and exit the program
    if k%256 == 27: 
            break
    #SPACE pressed: 
    elif k%256 == 32:       
            img_name = "facedetect_webcam_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            
# When everything is done, release the capture and close all the windows opened by open cv.
video_capture.release()
cv2.destroyAllWindows()