"""
@file treinar_modelo.py
@author João Pedro Arruda da Silva (joaosilva.2007@alunos.utfpr.edu.br)
@brief Treinamento de modelo de classificação de gestos com RandomForest
@version 0.1
@date 2026-09-05

@copyright Copyright (c) 2026
"""

# A saber: a vida eterna aos que, com perseverança em fazer bem, procuram glória, e honra, e incorrupção.
# Romanos 2:7

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('dados.csv', header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acuracia * 100:.2f}%")

joblib.dump(modelo, 'modelo_gestos.pkl')
print("Modelo salvo em 'modelo_gestos.pkl'")