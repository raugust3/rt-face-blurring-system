import cv2
import numpy as np

def detect_n_blur_faces(): 
    # load the pre-trained face detection classifier
    # this xml file contains the trained model for detecting frontal faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # initialize webcam - 0
    cap = cv2.VideoCapture(0)
    
    # check if the camera opened successfully. if not, print an error message and exit the function
    if not cap.isOpened():
        print("error: could not open camera - check if it's being used by another application")
        return
    
    print("camera started successfully - press 'q' to quit, 's' to save screenshot")
    
    while True:
        # capture frame-by-frame from the webcam
        # ret is a boolean that indicates if the frame was captured successfully
        # frame contains the actual image data from the camera
        ret, frame = cap.read()
        
        # if frame capture fails, break out of the loop
        if not ret:
            print("error: could not read frame from camera")
            break
        
        # convert the frame from color to grayscale
        # face detection works better and faster on grayscale image because it reduces computational complexity by removing color information
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect faces in the grayscale image using the haarcascade classifier
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # process each detected face in the current frame
        # faces is a list of rectangles where each rectangle is (x, y, width, height)
        for (x, y, w, h) in faces:
            # extract the region of interest (roi) - this is the area containing the face
            # the coordinates are: y to y+height (rows), x to x+width (columns)
            face_roi = frame[y:y+h, x:x+w]
            
            # apply gaussian blur to the face region
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)        
            
            # replace the original face region in the frame with the blurred version
            # this effectively blurs only the face while keeping the rest of the image clear
            frame[y:y+h, x:x+w] = blurred_face

            # draw bounding box around the blurred face
            # (x, y) is the top-left corner and (x+w, y+h) is the bottom-right
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        
        # display the processed frame in a window
        cv2.imshow('Real Time Face Blur Cam', frame)
        
        # wait for keyboard input (1 millisecond) and check for specific keys
        # the & 0xff part is used to get the last 8 bits of the key code
        key = cv2.waitKey(1) & 0xff
        
        # if 'q' is pressed, break out of the loop and exit
        if key == ord('q'):
            print("quitting application...")
            break
        # if 's' is pressed, save the current frame as a png image
        elif key == ord('s'):
            cv2.imwrite('blurred_faces_screenshot.png', frame)
            print("screenshot saved as 'blurred_faces_screenshot.png'")
    
    # cleanup: release the camera and destroy all opencv windows
    # important to free up the camera resource for other applications
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # let's run it
    detect_n_blur_faces()