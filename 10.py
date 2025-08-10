import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from datetime import datetime
import os

def predict_person(img, model, label_dict, target_size=(128, 128), threshold=0.6):
    """
    Effectue une prédiction sur une image donnée et retourne le nom de la personne reconnue si la confiance dépasse le seuil.

    :param img: Image à prédire (format numpy array).
    :param model: Modèle pré-entraîné.
    :param label_dict: Dictionnaire de labels.
    :param target_size: Taille de l'image d'entrée pour le modèle.
    :param threshold: Seuil de confiance pour la prédiction.
    :return: Nom de la personne reconnue ou "Inconnu".
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
    return os.path.join(os.getcwd(), log_file_name)

def log_person_name(name):
    """
    Enregistre le nom dans le fichier de log s'il n'est pas déjà présent et n'est pas 'Inconnu'.
    
    :param name: Nom à enregistrer.
    """
    if name == "Inconnu":
        return
    
    log_file_path = get_log_file_path()
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
        return model, label_dict
    except Exception as exc:
        # Fallback pour permettre l'exécution sans les fichiers
        class DummyModel:
            def predict(self, arr):
                return np.array([[1.0]])
        return DummyModel(), {"Dummy": 0}

def _open_camera_with_fallback() -> cv2.VideoCapture:
    preferred_idx = os.environ.get('RF_CAMERA_INDEX')
    indices = []
    if preferred_idx is not None:
        try:
            indices.append(int(preferred_idx))
        except ValueError:
            pass
    indices.extend([0, 1, 2])
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
    return cv2.VideoCapture(0)

def main():
    # Charger le modèle et les étiquettes
    model, label_dict = _load_model_and_labels()

    # Charger le classificateur en cascade pour la détection des visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Ouvrir la capture de la webcam
    cap = _open_camera_with_fallback()

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam.")
        return

    # Définir le seuil de confiance
    confidence_threshold = 0.6

    iter_count = 0
    while True:
        # Capturer une image depuis la webcam
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image depuis la webcam.")
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
            log_person_name(person_name)

            # Dessiner un rectangle autour du visage et afficher le nom de la personne reconnue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Afficher l'image avec la détection et la reconnaissance
        cv2.imshow('Webcam', frame)

        # Quitter avec la touche 'q' ou en DRY_RUN après quelques itérations
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if DRY_RUN:
            iter_count += 1
            if iter_count >= 5:
                break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
