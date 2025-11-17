# LSP-EnSeñas – Reconocimiento de señas en video

Este proyecto permite reconocer señas de la **Lengua de Señas Peruana (LSP)** a partir de videos, utilizando:

- **MediaPipe Holistic** para extraer landmarks del cuerpo y manos.
- **TensorFlow LSTM** para aprender el movimiento en el tiempo.
- **OpenCV** para capturar y procesar video.
- **Google Colab** para entrenamiento del modelo.
- **Python** para correr el modelo en tiempo real desde webcam.

---

### 🧠 ¿Qué hace?

- Aprende señas a partir de videos cortos.
- Procesa los movimientos en secuencia (video, no solo imágenes).
- Genera un modelo que identifica la seña con su probabilidad.

---

### 🔧 ¿Qué archivos incluye?

- `lsp_holistic_singtrad_LSTM.ipynb`: Notebook principal. Entrenas, evalúas y pruebas el modelo con tus videos.
- `sign_lstm_realtime.py`: Script para reconocimiento en vivo con webcam.
- Carpeta `SignProject/models/`:
  - `sign_model_lstm_v1.keras`: modelo entrenado en formato Keras.
  - `label_names.json`: lista de señas reconocidas por el modelo.

Todos estos archivos están en este Drive:

🔗 **Carpeta del proyecto (archivos listos para usar):**  
https://drive.google.com/drive/folders/1P367OVTz7mq8VLk544odZlAYIK1H9COv?usp=drive_link

---

### 🛠️ Cómo usar el Notebook

1. Abre `lsp_holistic_singtrad_LSTM.ipynb` en Google Colab.
2. Ejecuta las celdas paso a paso:
   - Montar el Drive.
   - Extraer landmarks con MediaPipe.
   - Crear dataset de secuencias.
   - Entrenar el modelo LSTM.
   - Evaluar y probar el modelo con tus propios videos.
3. El modelo final se guarda en `SignProject/models/`.

---

### 🖥️ Cómo reconocer señas en vivo

1. Asegúrate de tener Python 3.9+ y estas librerías:

pip install mediapipe opencv-python tensorflow numpy

2. Guarda la carpeta `SignProject/models/` junto al archivo `sign_lstm_realtime.py`.
3. Ejecuta el script:
python sign_lstm_realtime.py
4. En la ventana de cámara:
   - Haz una seña.
   - Presiona `R` para capturarla.
   - Se mostrará en pantalla la seña detectada con la confianza.

---

### ⚙️ Tecnologías usadas

- MediaPipe Holistic (3D pose + manos).
- TensorFlow (LSTM para secuencias).
- OpenCV (captura de video).
- Google Colab (entrenamiento).
- Python 3.9+
- Aumentación por espejado para control de mano dominante (opcional).

---

*Inteligencia Artificial aplicada a accesibilidad inclusiva.
