import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("model_new.h5")

# Load the Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
target_names = ["Sarthak", "Seema", "Hugh Jackman", "RDJ"]


# Function to preprocess the detected face image
def preprocess_face(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to match the model's expected sizing
    resized = cv2.resize(gray, (47, 62))
    # Normalize to match the training data
    normalized = resized / 255.0
    # Flatten the image
    flattened = normalized.flatten()
    # Reshape to be (1, 2914)
    reshaped = flattened.reshape(1, -1)
    return reshaped


# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Read frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region of interest
        roi_gray = gray[y : y + h, x : x + w]

        # Preprocess the face image
        try:
            roi_processed = preprocess_face(frame[y : y + h, x : x + w])

            # Predict the label using the loaded model
            prediction = model.predict(roi_processed)

            # Get the predicted label
            predicted_label = np.argmax(prediction)

            # Display the predicted label
            cv2.putText(
                frame,
                target_names[predicted_label],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        except Exception as e:
            print(f"Error: {e}")

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
