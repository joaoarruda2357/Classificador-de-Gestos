# Classificador de Gestos em Tempo Real

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-orange.svg)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-yellow.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)](https://opencv.org/)

Sistema de visão computacional e aprendizado de máquina para reconhecimento de gestos manuais em tempo real. O projeto utiliza o **MediaPipe** para a extração espacial de 21 *landmarks* (pontos de articulação da mão) e um modelo treinado via **scikit-learn** para classificar o gesto ao vivo através da webcam.

## Gestos Reconhecidos (Modelo Padrão)
O modelo pré-treinado é capaz de identificar os seguintes gestos:
* 🖐️ **Mão aberta**
* ✊ **Punho fechado**
* ✌️ **Paz** (Letra V)
* 👍 **Polegar pra cima** (Joinha)

## 📸 Demonstração dos Gestos

| Mão Aberta | Punho Fechado | Paz | Polegar pra Cima |
|:---:|:---:|:---:|:---:|
| ![Mão Aberta](docs/mao_aberta.jpg) | ![Punho Fechado](docs/punho_fechado.jpg) | ![Paz](docs/paz.jpg) | ![Polegar](docs/polegar.jpg) |

> **Nota:** O projeto foi construído de forma modular. Você pode facilmente usar os scripts inclusos para coletar seus próprios dados e ensinar novos gestos à IA!

## Demonstração
> *[Substitua este texto por um GIF ou vídeo curto demonstrando o código rodando em sua máquina]*

## 🛠️ Tecnologias Utilizadas
* **Python 3.12**
* **MediaPipe** (Rastreamento de mãos)
* **OpenCV** (Captura e processamento de vídeo)
* **scikit-learn** (Modelo de classificação Random Forest)
* **Pandas & NumPy** (Manipulação de dados)

---

## 🚀 Como Executar

### 1. Clonando o Repositório
Abra o seu terminal e clone o projeto:
```bash
git clone [https://github.com/arruda_/gesture-classifier](https://github.com/arruda_/gesture-classifier)
cd gesture-classifier

```

### 2. Configurando o Ambiente Virtual

É altamente recomendado o uso de um ambiente virtual (venv) para evitar conflitos de dependências:

```bash
# Criação do ambiente
python3 -m venv venv

# Ativação (Linux/macOS)
source venv/bin/activate
# No Windows, utilize: venv\Scripts\activate

# Instalação dos pacotes
pip install mediapipe opencv-python scikit-learn numpy joblib pandas

```

### 3. Baixando o Modelo Base do MediaPipe

O rastreador de mãos do MediaPipe exige um arquivo `.task` para rodar. Baixe-o com o comando abaixo:

```bash
wget -q [https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

```

*(Caso use Windows sem o `wget`, basta baixar o arquivo pelo navegador [neste link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) e salvá-lo na pasta raiz do projeto).*

### 4. Rodando a Demo

Com tudo pronto, inicie o reconhecimento em tempo real:

```bash
python3 demo.py

```

Pressione a tecla **'q'** na janela da câmera para encerrar a aplicação.

---

## 📂 Estrutura do Projeto

O projeto é dividido em etapas claras de Machine Learning. Siga a ordem abaixo caso queira treinar seus próprios gestos:

| Arquivo | Descrição |
| --- | --- |
| `visualizar.py` | Apenas exibe a imagem da webcam com o mapeamento dos *landmarks* (sem IA de classificação). Útil para testes de câmera. |
| `coletar.py` | Script de coleta. Permite gravar amostras do seu movimento e exportar as coordenadas tridimensionais para o dataset `dados.csv`. |
| `treinar.py` | Lê o `dados.csv`, treina o algoritmo de classificação e salva o modelo resultante em formato `.pkl`. |
| `demo.py` | Aplicação final. Junta a captura da câmera, a extração do MediaPipe e a predição do modelo treinado. |

---

**Autor:** João Pedro Arruda da Silva

```
