"""
@file coleta_dados.py
@author João Pedro Arruda da Silva (joaosilva.2007@alunos.utfpr.edu.br)
@brief Script para coleta de dados de landmarks das mãos via webcam
@version 0.1
@date 2026-09-05

@copyright Copyright (c) 2026
"""

# Não temas, ó pequeno rebanho, porque a vosso Pai agradou dar-vos o Reino.
# Lucas 12:32

import cv2
import mediapipe as mp
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--gesto', required=True, help='Nome do gesto sendo coletado')
args = parser.parse_args()

NUM_AMOSTRAS = 500

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
contador = 0

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

    for hand in result.hand_landmarks:
        for landmark in hand:
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        linha = []
        for landmark in hand:
            linha.extend([landmark.x, landmark.y, landmark.z])
        linha.append(args.gesto)

        with open('dados.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(linha)

        contador += 1

    texto = f"Gesto: {args.gesto} | {contador}/{NUM_AMOSTRAS}"
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Coleta de Dados", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if contador >= NUM_AMOSTRAS:
        print(f"Coleta concluída: {NUM_AMOSTRAS} amostras de '{args.gesto}' salvas em dados.csv")
        break

cap.release()
cv2.destroyAllWindows()