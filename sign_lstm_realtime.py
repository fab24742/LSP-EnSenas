import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

# ==========================
# CONFIG
# ==========================
MAX_FRAMES = 20   # MISMO VALOR QUE EN EL NOTEBOOK

BASE_DIR = os.path.join(os.getcwd(), "SignProject")
MODELS_DIR = os.path.join(BASE_DIR, "models")

model_path = os.path.join(MODELS_DIR, "sign_model_lstm_v1.keras")
labels_path = os.path.join(MODELS_DIR, "label_names.json")

print("Cargando modelo LSTM desde:", model_path)
model = keras.models.load_model(model_path)

with open(labels_path, "r") as f:
    label_names = json.load(f)

print("Clases:", label_names)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ==========================
# EXTRACCIÓN DE LANDMARKS
# (MISMA LÓGICA QUE ENTRENAMIENTO)
# ==========================
def extract_landmarks_from_results(results):
    """
    Convierte los resultados de MediaPipe Holistic en un vector 1D (225,)
    con pose (33), mano izq (21) y mano der (21).
    """
    def get_xyz(landmarks, n_points):
        if landmarks is None:
            data = [[0.0, 0.0, 0.0]] * n_points
        else:
            data = [[lm.x, lm.y, lm.z] for lm in landmarks]
            if len(data) < n_points:
                data += [[0.0, 0.0, 0.0]] * (n_points - len(data))
            data = data[:n_points]
        return data

    pose = get_xyz(results.pose_landmarks.landmark if results.pose_landmarks else None, 33)
    left_hand = get_xyz(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, 21)
    right_hand = get_xyz(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, 21)

    all_points = pose + left_hand + right_hand
    return np.array(all_points, dtype=np.float32).flatten()  # (225,)


def sequence_from_buffer(buffer, max_frames=MAX_FRAMES):
    """
    Convierte una lista de vectores (n_frames, 225)
    en una secuencia (1, max_frames, 225) con padding/corte.
    """
    arr = np.array(buffer, dtype=np.float32)  # (n_frames, 225)

    if arr.shape[0] < max_frames:
        pad_len = max_frames - arr.shape[0]
        pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:max_frames, :]

    arr = arr.reshape(1, max_frames, arr.shape[1])  # (1, T, 225)
    return arr


def predict_sequence(seq):
    """
    Recibe (1, T, 225) y devuelve label, conf, probs.
    """
    probs = model.predict(seq, verbose=0)[0]  # (num_classes,)
    idx = int(np.argmax(probs))
    label = label_names[idx]
    conf = float(probs[idx])
    return label, conf, probs


# ==========================
# LOOP PRINCIPAL WEBCAM
# ==========================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    recording = False
    frame_buffer = []
    last_label = None
    last_conf = 0.0

    print("\n=== SIGN LSTM REALTIME ===")
    print("Ventana de cámara:")
    print("  - Pulsa 'R' para empezar a grabar un gesto.")
    print("  - Vuelve a pulsar 'R' para terminar y ver la predicción.")
    print("  - Pulsa 'ESC' para salir.\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Volteamos horizontal para estilo espejo
            frame = cv2.flip(frame, 1)

            # Procesar con MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # (Opcional) dibujar landmarks para el profe
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            # Si estamos grabando, guardamos el vector de ese frame
            if recording:
                vec = extract_landmarks_from_results(results)
                frame_buffer.append(vec)

            # Dibujar UI bonita
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

            cv2.putText(
                frame, "Sign LSTM - Demo",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if recording:
                txt = "GRABANDO gesto... (R para terminar)"
                color = (0, 0, 255)
            else:
                txt = "Pulsa R para grabar un gesto"
                color = (255, 255, 255)

            cv2.putText(
                frame, txt,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if last_label is not None:
                cv2.putText(
                    frame,
                    f"Ultima prediccion: {last_label} ({last_conf:.2f})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Sign LSTM - Webcam", frame)

            key = cv2.waitKey(1) & 0xFF

            # ESC -> salir
            if key == 27:
                break

            # R/r -> toggle grabación
            elif key in (ord('r'), ord('R')):
                if not recording:
                    # Empezar nueva secuencia
                    frame_buffer = []
                    recording = True
                    print("\n=== GRABANDO GESTO... ===")
                else:
                    # Terminar y predecir
                    recording = False
                    if len(frame_buffer) == 0:
                        print("⚠ No se capturaron frames, intenta de nuevo.")
                        continue

                    seq = sequence_from_buffer(frame_buffer, MAX_FRAMES)
                    label, conf, probs = predict_sequence(seq)
                    last_label, last_conf = label, conf

                    print("\n=== RESULTADO DEL GESTO ===")
                    print(f"Prediccion: {label} (confianza {conf:.2f})")
                    print("Distribucion de probabilidades:")

                    # ordenar de mayor a menor probabilidad
                    indices = np.argsort(probs)[::-1]
                    for i in indices:
                        print(f"  {label_names[i]}: {probs[i]:.2f}")
                    print("===========================\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
