# LSP-EnSeñas – Reconocimiento de señas con MediaPipe + LSTM

LSP-EnSeñas es un sistema de reconocimiento de señas basado en:

- **MediaPipe Holistic** → extrae landmarks (cuerpo + manos)
- **LSTM (Long Short-Term Memory)** → entiende movimiento en el tiempo
- **TensorFlow / Keras** → entrena y ejecuta el modelo
- **OpenCV** → captura video (tanto en Colab como en webcam local)

Este proyecto permite:

✔ Entrenar un modelo a partir de videos organizados por clases  
✔ Evaluarlo dentro del notebook  
✔ Exportarlo listo para inferencia  
✔ Ejecutarlo en tu computadora por webcam en tiempo real  

---

## 📁 Estructura del repositorio

