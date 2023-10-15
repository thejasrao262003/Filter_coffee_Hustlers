
import dlib
import cv2
import os, time
import matplotlib.pyplot as plt
from face_rec_img import face_recogntion
output_directory = "Detected_faces"
os.makedirs(output_directory, exist_ok=True)
import shutil
if os.path.exists("/Users/tejas/PycharmProjects/pythonProject/Final_Project/Detected_faces"):
    shutil.rmtree("/Users/tejas/PycharmProjects/pythonProject/Final_Project/Detected_faces")

if not os.path.exists("/Users/tejas/PycharmProjects/pythonProject/Final_Project/Detected_faces"):
    os.makedirs("./Detected_faces")
from emotionDetector import emotion_detection
import pandas as pd
row_name = []
for i in os.listdir("Images"):
    row_name.append((i[:i.index(".")]).upper())

column_names = ["tot"]#["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
lis = [[0]*len(column_names) for _ in range(len(row_name))]
df = pd.DataFrame(lis, columns=column_names, index=row_name)
weights = {"Angry": 0.1, "Disgusted": 0.1, "Fearful": 0.9, "Happy": 0.3, "Neutral": 0.9, "Sad": 0.4, "Surprised": 0.6}
wght_tot = (sum(weights.values())/len(weights))
df2 = pd.DataFrame(lis, columns=column_names, index=row_name)


TIMELINE = []
students ={"SHISHIR":0, "SHRIHARI":0, "THEJAS":0, "VIKAS": 0}
def face_splitter():
    face_detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    time_in_s = 10
    n_for_agg = 5
    curr_time = time.time()

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
        # [cv2.imshow("grey_scale", x) for x in faces]
        face_count = 0
        if (t := time.time()) > curr_time + time_in_s:
            if not os.path.exists("Detected_faces"):
                os.makedirs("Detected_faces")
            for face in faces:
                x, y, w, h = map(lambda x: max(x,0),[face.left(), face.top(), face.width(), face.height()])
                # print(f"{x},{y},{w},{h}::::")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_image = frame[y:y + h, x:x + w]
                face_filename = os.path.join(output_directory, f"face_{curr_time + time_in_s}_{face_count}.png")

                cv2.imwrite(face_filename, face_image)
                face_count += 1
            curr_time = t
            dict = face_recogntion()
            dict = emotion_detection(dict)
            shutil.rmtree("Detected_faces")
            # print(dict)
            print(f"{dict=}")
            for v in dict.values():
                    df.at[v[0], "tot"] += weights[v[1]]
            # print(df)
        frame_count += 1
        if frame_count%n_for_agg == 0:
            sprint = df['tot'].sum()/(n_for_agg*len(row_name)*wght_tot**2)
            TIMELINE.append(sprint)
            for name in row_name:
                if (df.at[name, "tot"] / (n_for_agg * wght_tot)) > sprint:
                    students[name] += 1
                else:
                    students[name] += 0
            plt.plot((*range(len(TIMELINE)),), TIMELINE)

            df.loc[:, :] = 0
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

face_splitter()
print(TIMELINE)
print(students)
plt.show()
for k,v in students.items():
    print(f"{k} payed attenttion for {v*100/len(TIMELINE)} % of the class")

