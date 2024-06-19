import cv2
import os


# Define the face extractor function
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Crop all faces found
    for x, y, w, h in faces:
        cropped_face = img[y : y + h, x : x + w]
        return cropped_face


# Paths
input_folder = "Dataset\Train\Robert Downey Jr"
output_folder = "Images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Loop through images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Ensure correct file types
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            face = face_extractor(img)
            if face is not None:
                # Save the extracted face to the output folder
                face_filename = os.path.join(output_folder, filename)
                cv2.imwrite(face_filename, face)
                print(f"Saved {face_filename}")
            else:
                print(f"No face found in {filename}")
        else:
            print(f"Could not read {filename}")
