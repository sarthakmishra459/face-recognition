import cv2

stream_url = "udp://192.168.159.1:8080"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Failed to open video stream")
else:
    print("Video stream opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
