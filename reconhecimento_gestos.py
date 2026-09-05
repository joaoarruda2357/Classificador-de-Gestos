"""
@file reconhecimento_gestos.py
@author João Pedro Arruda da Silva (joaosilva.2007@alunos.utfpr.edu.br)
@brief Reconhecimento de gestos em tempo real usando MediaPipe e RandomForest
@version 0.1
@date 2026-09-05

@copyright Copyright (c) 2026
"""

# Depois disso, ouvi a voz do Senhor, que dizia: A quem enviarei, e quem há de ir por nós? Então, disse eu: eis-me aqui, envia-me a mim.
# Isaías 6:8

import cv2
import mediapipe as mp
import joblib
import numpy as np

modelo = joblib.load('modelo_gestos.pkl')

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Não consegui ler o frame da câmera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    gesto_atual = "Nenhuma mao detectada"

    for hand in result.hand_landmarks:
        for landmark in hand:
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        features = []
        for landmark in hand:
            features.extend([landmark.x, landmark.y, landmark.z])

        features = np.array(features).reshape(1, -1)
        gesto_atual = modelo.predict(features)[0]

    cv2.putText(frame, gesto_atual, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Reconhecimento de Gestos", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()