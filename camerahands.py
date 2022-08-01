import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model = load_model('signlanguage.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

# def bbox(pic):
#     framergb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
#     result = hands.process(framergb)
#     hand_landmarks = result.multi_hand_landmarks
#     if hand_landmarks:
#         for handLMs in hand_landmarks:
#             x_max = 0
#             y_max = 0
#             x_min = w
#             y_min = h
#             for lm in handLMs.landmark:
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 if x > x_max:
#                     x_max = x
#                 if x < x_min:
#                     x_min = x
#                 if y > y_max:
#                     y_max = y
#                 if y < y_min:
#                     y_min = y
#             y_min -= 20
#             y_max += 20
#             x_min -= 20
#             x_max += 20


img_counter = 0
analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        analysisframe = frame
        showframe = analysisframe

        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        cv2.imshow("Frame", showframe)
        analysisframe = cv2.resize(analysisframe,(128,128))
        analysisframe = np.reshape(analysisframe,[1,128,128,3])
        analysisframe = analysisframe/255.0
        prediction = model.predict(analysisframe)
        predarray = np.array(prediction[0])
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key,value in letter_prediction_dict.items():
            if value==high1:
                print("Predicted Letter 1: ", key)
                print('Confidence 1: ', 100*value)
            elif value==high2:
                print("Predicted Letter 2: ", key)
                print('Confidence 2: ', 100*value)
            elif value==high3:
                print("Predicted Letter 3: ", key)
                print('Confidence 3: ', 100*value)
        time.sleep(5)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()