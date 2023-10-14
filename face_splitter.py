import cv2
import os

output_directory = "detected_faces"
os.makedirs(output_directory, exist_ok=True)
running = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
itteration = 0

while running:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    face_count = 0
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)        
        face_image = frame[y:y + h, x:x + w]
        face_filename = os.path.join(output_directory, f"face_{itteration}_{face_count}.png")
        cv2.imwrite(face_filename, face_image)
        face_count += 1
    itteration += 1

    # Display the video stream with detected faces
    cv2.imshow('All Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
