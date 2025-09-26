import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import json

# ===== Configuration =====
GESTURE_MAPPING = {
    'play': 'one_finger',
    'next': 'two_fingers',
    'previous': 'three_fingers',
    'volume_down': 'four_fingers',
    'volume_up': 'five_fingers',
    'pause': 'closed_fist'
}

IMG_SIZE = 96  # Reduced for faster CPU processing
BATCH_SIZE = 32  # Reduced for CPU memory limitations
EPOCHS = 50
LEARNING_RATE = 0.001

# ===== Optimized Data Loader =====
class OptimizedGestureLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.label_names = list(GESTURE_MAPPING.keys())
        self.label_to_int = {label: i for i, label in enumerate(self.label_names)}
        
        # Verify dataset structure
        print("🔍 Verifying dataset structure...")
        for folder in GESTURE_MAPPING.values():
            folder_path = os.path.join(dataset_path, folder)
            if not os.path.exists(folder_path):
                print(f"❌ Missing folder: {folder_path}")
            else:
                num_files = len([f for f in os.listdir(folder_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"✅ {folder}: {num_files} images")

    def _fast_preprocess(self, img_path):
        """Optimized image loading and preprocessing"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Could not read: {img_path}")
                return None
            
            # Fast processing pipeline
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            return img.astype(np.float32) / 255.0
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {str(e)}")
            return None

    def load_dataset(self):
        """Load and balance dataset"""
        X, y = [], []
        
        for label, folder in GESTURE_MAPPING.items():
            folder_path = os.path.join(self.dataset_path, folder)
            if not os.path.exists(folder_path):
                continue
                
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                processed = self._fast_preprocess(img_path)
                if processed is not None:
                    X.append(processed)
                    y.append(self.label_to_int[label])
        
        if not X:
            print("❌ No images were loaded! Please check:")
            print("1. Dataset path is correct")
            print("2. Images exist in subfolders")
            print("3. Images are in .jpg, .png or .jpeg format")
            return None, None
            
        X = np.expand_dims(np.array(X), axis=-1)
        y = np.array(y)
        print(f"\n📊 Successfully loaded {len(X)} images")
        return X, y

# ===== Lightweight Model =====
def create_fast_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===== Main Function =====
def main():
    # Initialize loader
    dataset_path = input("Enter dataset path (or press Enter for 'dataset'): ").strip() or 'dataset'
    loader = OptimizedGestureLoader(dataset_path)
    
    # Load data
    X, y = loader.load_dataset()
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create optimized data pipeline
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_data = val_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Create and train model
    model = create_fast_model((IMG_SIZE, IMG_SIZE, 1), len(loader.label_names))
    
    print("\n🚀 Starting training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ],
        verbose=1
    )
    
    # Save results
    model.save('gesture_model.h5')
    with open('model_info.json', 'w') as f:
        json.dump({
            'gesture_mapping': GESTURE_MAPPING,
            'input_shape': [IMG_SIZE, IMG_SIZE, 1],
            'classes': loader.label_names,
            'accuracy': float(max(history.history['val_accuracy']))
        }, f)
    
    print("\n✅ Training complete! Saved:")
    print("- gesture_model.h5")
    print("- model_info.json")

if __name__ == "__main__":
    main()