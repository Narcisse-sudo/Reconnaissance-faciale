import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime

# Fonction pour charger et étiqueter les images depuis les sous-dossiers
def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        label_dict[person_name] = current_label
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)
                labels.append(current_label)
        
        current_label += 1
    
    return np.array(images), np.array(labels), label_dict

# Fonction pour normaliser les images
def preprocess_images(images):
    return images / 255.0

# Fonction pour obtenir le répertoire des logs pour TensorBoard
def get_tensorboard_log_dir():
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join(os.getcwd(), "logs", f"fit_{today_date}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Partie principale du script
if __name__ == "__main__":
    # Charger les images de formation
    train_root = os.environ.get('RF_TRAIN_DIR', os.path.join(os.getcwd(), 'data', 'train'))
    train_images, train_labels, train_label_dict = load_images_from_folder(train_root)
    
    # Charger les images de validation
    val_root = os.environ.get('RF_VAL_DIR', os.path.join(os.getcwd(), 'data', 'val'))
    val_images, val_labels, val_label_dict = load_images_from_folder(val_root)

    # Assurer que le dictionnaire des labels est le même pour la formation et la validation
    assert train_label_dict == val_label_dict, "Les dictionnaires de labels doivent être identiques pour la formation et la validation."

    # Prétraiter les images
    train_images = preprocess_images(train_images)
    val_images = preprocess_images(val_images)

    # Définir le modèle CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(train_label_dict), activation='softmax')
    ])

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Configurer TensorBoard
    log_dir = get_tensorboard_log_dir()
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Entraîner le modèle
    model.fit(train_images, train_labels, epochs=100, batch_size=32, validation_data=(val_images, val_labels), callbacks=[tensorboard_callback])

    # Sauvegarder le modèle et les étiquettes
    model.save('face_recognition_model.h5')  # Sauvegarder le modèle
    np.save('label_dict.npy', train_label_dict)  # Sauvegarder les étiquettes

    print(f"Modèle sauvegardé sous 'face_recognition_model.h5'")
    print(f"Étiquettes sauvegardées sous 'label_dict.npy'")
    print(f"Les logs de TensorBoard sont dans '{log_dir}'")
