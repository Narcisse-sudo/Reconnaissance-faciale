import os
import cv2

# Chemin du dossier contenant les images à détecter
# Par défaut, utiliser un chemin relatif au projet si les chemins absolus n'existent pas
default_input = os.path.join(os.getcwd(), 'raw', 'Inconnu')
default_output = os.path.join(os.getcwd(), 'data', 'raw', 'Inconnu_face')

input_folder = os.environ.get('RF_INPUT_DIR', default_input)
# Dossier où les images des visages seront enregistrées
output_folder = os.environ.get('RF_OUTPUT_DIR', default_output)
 
# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Charger le classifieur en cascade pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def main() -> None:
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                    print(f"Created output subfolder: {output_subfolder}")
                image_to_detect = cv2.imread(image_path)
                if image_to_detect is None:
                    print(f"Erreur lors du chargement de l'image {image_path}")
                    continue
                gray_image = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                print(f'There are {len(faces)} faces in the image {file}')
                if len(faces) == 0:
                    print(f"No faces detected in image {file}")
                for index, (x, y, w, h) in enumerate(faces):
                    print(f'Found face {index+1} at x:{x}, y:{y}, w:{w}, h:{h} in file {file}')
                    current_face_image = image_to_detect[y:y+h, x:x+w]
                    face_image_path = os.path.join(output_subfolder, f'{os.path.splitext(file)[0]}_face_{index+1}.jpg')
                    cv2.imwrite(face_image_path, current_face_image)
                    print(f"Saving face image to: {face_image_path}")


if __name__ == "__main__":
    main()
