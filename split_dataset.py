import os
import shutil
import random

# Paths
source_folder = 'gesture_dataset'
train_folder = 'gesture_dataset_split/train'
val_folder = 'gesture_dataset_split/validation'

# Split ratio
split_ratio = 0.8  # 80% training, 20% validation

# Create destination folders
for folder in [train_folder, val_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Go through each gesture class
for class_name in os.listdir(source_folder):
    class_path = os.path.join(source_folder, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)
        split_point = int(split_ratio * len(images))

        train_images = images[:split_point]
        val_images = images[split_point:]

        # Create class folders in train/val
        os.makedirs(os.path.join(train_folder, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_folder, class_name), exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_folder, class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_folder, class_name, img))

print("✅ Dataset split successfully!")
