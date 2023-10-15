
import os

import cv2
import numpy as np
from keras.models import model_from_json

def emotion_detection(dict):
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        weights = {"Angry":0.1, "Disgusted":0.1, "Fearful":0.9, "Happy":0.3, "Neutral":0.9, "Sad":0.4, "Surprised":0.6}
        # load json and create model
        json_file = open('emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(loaded_model_json)

        # load weights into new model
        emotion_model.load_weights("emotion_model.h5")
        print("Loaded model from disk")
        print(os.listdir("Detected_faces"))
        for i in os.listdir("Detected_faces"):
                print(f"Detected_faces/{i}")
                cap = cv2.imread(f"Detected_faces/{i}")
                frame = cv2.resize(cap, (1280, 720))
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                if dict.get(i)is not None: dict[i].append(emotion_dict[maxindex])



        cv2.destroyAllWindows()
        return dict
