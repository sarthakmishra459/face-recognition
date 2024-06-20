import cv2
import numpy as np
from keras.models import load_model
import sys

# Load the saved model
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
model = load_model("model_new.h5")

# Loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
persons = ["Sarthak", "Seema", "Hugh Jackman", "RDJ"]


def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None, None  # Return None and None if no face detected

    # Assuming only one face in webcam scenario, crop the first face found
    (x, y, w, h) = faces[0]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cropped_face = img[y : y + h, x : x + w]

    return cropped_face, (x, y, w, h)  # Return cropped face and its coordinates


# Doing Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()

    # Preprocess the captured frame to get face
    face, (x, y, w, h) = face_extractor(frame)

    if face is not None:
        try:
            # Resize the face to match model input size (62x47) and convert to grayscale
            face_resized = cv2.resize(face, (62, 47))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

            # Normalize grayscale image (0-255) to values (0-1)
            gray = gray / 255

            # Reshape the image to match model input shape
            roi = gray.reshape(1, -1)

            # Predict using the loaded model
            result = model.predict(roi)

            # Get the predicted label index
            predicted_class = np.argmax(result)

            # Get the corresponding label name
            label_name = persons[predicted_class]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Determine the recognized name based on predictions
            name = label_name

            # Display the recognized name on the frame
            cv2.putText(
                frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )

        except Exception as e:
            print(f"Exception: {e}")
            cv2.putText(
                frame,
                "No Face Found",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            pass

    else:
        cv2.putText(
            frame,
            "No face found",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
