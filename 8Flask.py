import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from datetime import datetime
import os
from flask import Flask, Response, render_template

# Initialiser Flask
app = Flask(__name__)

DRY_RUN = os.environ.get("RF_DRY_RUN", "0") == "1"

def _load_model_and_labels():
    if DRY_RUN:
        class DummyModel:
            def predict(self, arr):
                return np.array([[1.0]])
        return DummyModel(), {"Dummy": 0}
    try:
        model = models.load_model('face_recognition_model.h5')
        label_dict = np.load('label_dict.npy', allow_pickle=True).item()
        print("Modèle et labels chargés avec succès.")
        return model, label_dict
    except Exception as exc:
        print(f"[AVERTISSEMENT] Échec du chargement du modèle/labels: {exc}. Utilisation d'un modèle fictif (Dummy).")
        class DummyModel:
            def predict(self, arr):
                return np.array([[1.0]])
        return DummyModel(), {"Dummy": 0}

# Charger le modèle et les étiquettes
model, label_dict = _load_model_and_labels()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Seuil de confiance
confidence_threshold = 0.6

def predict_person(img, model, label_dict, target_size=(128, 128), threshold=0.6):
    """
    Effectue une prédiction sur une image donnée et retourne le nom de la personne reconnue si la confiance dépasse le seuil.
    """
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)
    
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]
    
    if confidence >= threshold:
        for person_name, label in label_dict.items():
            if label == predicted_label:
                return person_name
    return "Inconnu"

def get_log_file_path():
    """
    Retourne le chemin du fichier de log basé sur la date du jour.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"recognized_faces_{today_date}.txt"
    return os.path.join(os.getcwd(), 'static', log_file_name)

def log_person_name(name):
    """
    Enregistre le nom dans le fichier de log s'il n'est pas déjà présent et n'est pas 'Inconnu'.
    """
    if name in ("Inconnu", "Dummy"):
        return
    
    log_file_path = get_log_file_path()
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as file:
            file.write(name + '\n')
    else:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
        
        # Éviter les doublons
        names = [line.strip() for line in lines]
        if name not in names:
            with open(log_file_path, 'a') as file:
                file.write(name + '\n')

def _open_camera_with_fallback() -> cv2.VideoCapture:
    """Try multiple camera indices and API backends until one opens.
    You can force an index with env RF_CAMERA_INDEX.
    """
    preferred_idx = os.environ.get('RF_CAMERA_INDEX')
    indices = []
    if preferred_idx is not None:
        try:
            indices.append(int(preferred_idx))
        except ValueError:
            pass
    indices.extend([0, 1, 2])
    # Remove duplicates while preserving order
    seen = set()
    indices = [i for i in indices if not (i in seen or seen.add(i))]

    backends = [getattr(cv2, 'CAP_DSHOW', None), getattr(cv2, 'CAP_MSMF', None), getattr(cv2, 'CAP_ANY', 0)]
    backends = [b for b in backends if b is not None]

    for idx in indices:
        for api in backends:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                print(f"Webcam ouverte: index={idx}, api={api}")
                return cap
            cap.release()
    raise RuntimeError("Impossible d'ouvrir la webcam (essayé indices 0/1/2 et backends DSHOW/MSMF/ANY).")

def generate_frames():
    """
    Génère des frames pour le streaming vidéo.
    """
    cap = _open_camera_with_fallback()
    
    while True:
        # Capturer une image depuis la webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir l'image en niveaux de gris pour la détection des visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extraire le visage détecté
            face_img = frame[y:y+h, x:x+w]
            # Effectuer la prédiction avec le seuil de confiance
            person_name = predict_person(face_img, model, label_dict, threshold=confidence_threshold)

            # Enregistrer le nom dans le fichier texte si ce n'est pas "Inconnu"
            if person_name != "Inconnu":
                log_person_name(person_name)

            # Dessiner un rectangle autour du visage et afficher le nom de la personne reconnue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convertir l'image en format JPEG pour le streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        # Générer des frames pour le streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """
    Route principale qui sert la page d'accueil avec le streaming vidéo.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Route pour le streaming vidéo en temps réel.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__' and not DRY_RUN:
    app.run(debug=True)
