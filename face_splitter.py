import dlib
import cv2
import os, time

output_directory = "detected_faces"
os.makedirs(output_directory, exist_ok=True)

face_detector = dlib.get_frontal_face_detector() #! dlib doesn't seem to work for me(RockerBot)
cap = cv2.VideoCapture(0)

time_in_s = 10
curr_time =0

frame_count = 0
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        running= False
        print("Stream Broken")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    face_count = 0
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_image = frame[y:y + h, x:x + w]
        face_filename = os.path.join(output_directory, f"face_{frame_count}_{face_count}.png")
        if (t:=time.time())>curr_time+time_in_s:
            cv2.imwrite(face_filename, face_image)
            face_count += 1
            curr_time = t
    frame_count +=1
    
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running=False
        break

cap.release()
cv2.destroyAllWindows()
