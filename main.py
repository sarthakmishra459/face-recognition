# sudo su
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
import cv2
import numpy as np


# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None

    # Detect faces
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y+h+50, x:x+w+50]
        return cropped_face  # Return the first face found

# Initialize Video Capture (use a video file instead of webcam for testing)
cap = cv2.VideoCapture('./video.mp4')
count = 0

# Collect 100 samples of your face from video input
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video file

    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = './Images/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
