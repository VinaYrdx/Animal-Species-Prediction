import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. SETUP & DATA PATH
# ==========================================
# IMPORTANT: Change this name to the data set folder, ive added this one for the reference
DATASET_PATH = 'raw-img' 

# Define constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10 # Set to 10 for a good balance of time/accuracy

# ==========================================
# 2. DATA AUGMENTATION & GENERATORS
# ==========================================
print("Loading and preprocessing data...")

# Main Data Generator with a 20% validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserves 20% of images for validation
)

# Training Set Generator
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation Set Generator
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Capture class names for later predictions
class_names = list(train_generator.class_indices.keys())
print(f"Detected classes: {class_names}")

# ==========================================
# 3. MODEL ARCHITECTURE (VGG-16)
# ==========================================
print("\nBuilding the VGG-16 Model...")

# Load VGG-16 base without top classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the base layers so we don't destroy pre-trained features
for layer in base_model.layers:
    layer.trainable = False

# Build custom head for our 10 specific classes
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 4. TRAINING THE MODEL
# ==========================================
print("\nStarting training...")

# Early stopping stops training if validation loss doesn't improve for 3 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop]
)

# ==========================================
# 5. VISUALIZING THE OUTPUT

print("\nPlotting training results...")

# Plot Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==========================================
# 6. TEST PREDICTION ON A SINGLE IMAGE

print("\n--- Testing Model Prediction ---")
# Let's pick the very first image from the validation set to test
test_img_path = validation_generator.filepaths[0]
actual_label = test_img_path.split(os.path.sep)[-2] # Extracts folder name

# Load and format the image
img = load_img(test_img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0 # Normalize just like we did in ImageDataGenerator

# Prediction : 
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
predicted_label = class_names[predicted_class_idx]
confidence = predictions[0][predicted_class_idx] * 100

print(f"\nTested on image: {test_img_path}")
print(f"Actual Animal: {actual_label}")
print(f"Predicted Animal: {predicted_label} (Confidence: {confidence:.2f}%)")