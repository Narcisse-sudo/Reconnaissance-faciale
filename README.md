# Reconnaissance faciale 

Application de reconnaissance faciale (OpenCV + TensorFlow/Keras) avec interface web Flask pour le streaming vidéo et la journalisation des personnes reconnues.

## Prérequis
- Windows 10/11 (PowerShell recommandé)
- Python 3.11 x64
- Webcam fonctionnelle

## Installation rapide
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Données et entraînement
- Arborescence attendue:
  - `data/train/<Nom_Personne>/*.jpg`
  - `data/val/<Nom_Personne>/*.jpg`
- Entraîner le modèle et générer les fichiers à la racine:
```powershell
python 3-1train_model.py
```
Résultats: `face_recognition_model.h5` et `label_dict.npy`.

Variables d’environnement (optionnelles):
- `RF_SOURCE_DIR`, `RF_TRAIN_DIR`, `RF_VAL_DIR` pour surcharger les chemins des datasets

## Lancer l’application web
```powershell
python 8Flask.py
```
Puis ouvrir `http://127.0.0.1:5000`.

## Scripts inclus
- `8Flask.py` — application web (streaming + logs)
- `10.py` — test webcam en console
- `2recuperationface.py` — extraction de visages dans des images
- `2split_dataset.py` — split train/val
- `1bwebcam2.py` — captures rapides de la webcam

## Tests automatiques (CI locale)
Exécute des imports en mode DRY_RUN pour vérifier que tout est exécutable sans matériel ni modèle:
```powershell
python run_tests.py
```

## Évaluation du modèle
Mesure l'accuracy, le rapport precision/recall/F1 par classe et génère une matrice de confusion:
```powershell
python evaluate_model.py
```
Sorties:
- Résultats chiffrés dans le terminal
- Image: `static/confusion_matrix.png`

## Journalisation
Les personnes reconnues du jour sont enregistrées dans `static/recognized_faces_YYYY-MM-DD.txt`.

## Fallback & variables utiles
- Fallback caméra: essai automatique indices 0/1/2 et backends DirectShow/MSMF. Pour forcer un index:
  - PowerShell: `$env:RF_CAMERA_INDEX=1; python 8Flask.py`
- Mode DRY_RUN (désactive webcam/modèle pour tests):
  - PowerShell: `$env:RF_DRY_RUN=1; python 8Flask.py`
  - Désactivation: `Remove-Item Env:RF_DRY_RUN -ErrorAction SilentlyContinue`

## Dépannage
- « Dummy » affiché à l’écran: le modèle n’a pas été chargé. Vérifier:
  - Présence de `face_recognition_model.h5` et `label_dict.npy` à la racine
  - Installer `h5py`: `pip install h5py`
  - Lancer ensuite: `python 8Flask.py`
- Webcam noire / occupée:
  - Fermer Teams/Zoom/navigateur utilisant la caméra
  - Essayer `RF_CAMERA_INDEX=1` (ou 2)
- Installation TensorFlow:
  - Le projet utilise `tensorflow-cpu==2.12.0` (compatible Python 3.11). Si souci GPU, rester sur la version CPU.
