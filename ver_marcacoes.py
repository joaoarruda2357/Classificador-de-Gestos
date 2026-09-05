"""
@file ver_marcacoes.py
@author João Pedro Arruda da Silva (joaosilva.2007@alunos.utfpr.edu.br)
@brief Rastreamento da mão usando MediaPipe e OpenCV
@version 0.1
@date 2026-09-05

@copyright Copyright (c) 2026
"""

# Alegrem-se na esperança, sejam pacientes na tribulação, perseverem na oração.
# Romanos 12:12

import cv2
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Não consegui ler o frame da câmera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    h, w, _ = frame.shape

    for hand in result.hand_landmarks:
        for landmark in hand:
            cx = int(landmark.x * w)  # calcula o pixel X a partir de landmark.x e w
            cy = int(landmark.y * h)  # calcula o pixel Y a partir de landmark.y e h
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Marcações da Mão", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()