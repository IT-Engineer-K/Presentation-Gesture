import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import time
import random

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)#カメラのIDを選ぶ
x_train = []
y_train = []
firstEsc = True
model =  tf.keras.models.load_model('model.h5')
a = 0
print("Start")
while True:
    success, image = cap.read()#キャプチャが成功していたら画像データとしてimageに取り込む
    #image = cv2.flip(image, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))#検出ポイントの名前のリスト

    data = []
    try:
        if len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            predictAble = True
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                if 0<x<1 and 0<y<1:
                    data.append(hand_landmarks.landmark[i].x)
                    data.append(hand_landmarks.landmark[i].y)
                    data.append(hand_landmarks.landmark[i].z)
                else:
                    predictAble = False
                    break
            if predictAble:
                predicted = model(np.array((data,)))[0]
                print(predicted)
                if predicted[0] > predicted[1]:
                    pyautogui.press('down')
                    # 誤作動が起きた時に改善するための画像を記録
                    cv2.imwrite('img/{}.png'.format(random.randint(0, 1000000000)), image)
                    time.sleep(1.5)
    except:
        pass
    if True:
        cv2.imshow('', image)
        if cv2.waitKey(1) != -1:
            break