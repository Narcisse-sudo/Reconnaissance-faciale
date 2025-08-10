import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    for user in os.listdir(source_dir):
        user_dir = os.path.join(source_dir, user)
        if os.path.isdir(user_dir):
            images = [f for f in os.listdir(user_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)

            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            val_images = images[split_index:]

            os.makedirs(os.path.join(train_dir, user), exist_ok=True)
            os.makedirs(os.path.join(val_dir, user), exist_ok=True)

            for image in train_images:
                shutil.copy(os.path.join(user_dir, image), os.path.join(train_dir, user, image))
            for image in val_images:
                shutil.copy(os.path.join(user_dir, image), os.path.join(val_dir, user, image))

def main() -> None:
    project_root = os.getcwd()
    source_dir = os.environ.get('RF_SOURCE_DIR', os.path.join(project_root, 'data', 'raw'))
    train_dir = os.environ.get('RF_TRAIN_DIR', os.path.join(project_root, 'data', 'train'))
    val_dir = os.environ.get('RF_VAL_DIR', os.path.join(project_root, 'data', 'val'))
    split_data(source_dir, train_dir, val_dir)

if __name__ == "__main__":
    main()
