import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tensorflow.keras import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_images_and_labels(
    root_dir: str,
    person_to_label: Dict[str, int],
    target_size: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    missing_classes: List[str] = []

    for person_name in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        if person_name not in person_to_label:
            missing_classes.append(person_name)
            continue

        label_index = person_to_label[person_name]
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(label_index)

    if missing_classes:
        print(
            f"[INFO] {len(missing_classes)} classe(s) dans val n'existent pas dans label_dict et ont été ignorées: "
            + ", ".join(missing_classes[:10])
            + (" ..." if len(missing_classes) > 10 else "")
        )

    if not images:
        raise RuntimeError("Aucune image chargée depuis le dossier de validation.")

    return np.array(images), np.array(labels), sorted(person_to_label, key=person_to_label.get)


def preprocess(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    plt.figure(figsize=(max(8, len(class_names) * 0.5), max(6, len(class_names) * 0.5)))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("Vérité terrain")
    plt.xlabel("Prédiction")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Matrice de confusion sauvegardée: {output_path}")


def main() -> None:
    project_root = os.getcwd()
    val_root = os.environ.get("RF_VAL_DIR", os.path.join(project_root, "data", "val"))

    model_path = os.path.join(project_root, "face_recognition_model.h5")
    labels_path = os.path.join(project_root, "label_dict.npy")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            "Modèle ou labels manquants. Assurez-vous que 'face_recognition_model.h5' et 'label_dict.npy' sont à la racine."
        )

    print("[INFO] Chargement du modèle et des labels...")
    model = models.load_model(model_path)
    person_to_label: Dict[str, int] = np.load(labels_path, allow_pickle=True).item()

    print(f"[INFO] Chargement des images de validation depuis: {val_root}")
    val_images, val_labels, ordered_class_names = load_images_and_labels(val_root, person_to_label)

    val_images = preprocess(val_images)

    print("[INFO] Inférence en cours...")
    preds = model.predict(val_images, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(val_labels, y_pred)
    print(f"\n=== Résultats ===\nAccuracy (val): {acc:.4f}")

    print("\nRapport par classe:")
    print(
        classification_report(
            val_labels,
            y_pred,
            target_names=ordered_class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(val_labels, y_pred, labels=list(range(len(ordered_class_names))))
    cm_path = os.path.join(project_root, "static", "confusion_matrix.png")
    save_confusion_matrix(cm, ordered_class_names, cm_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERREUR] {exc}")
        sys.exit(1)

