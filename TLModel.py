import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# DATA SETUP
split_base_dir = ""
train_dir = os.path.join(split_base_dir, "train")
val_dir = os.path.join(split_base_dir, "val")
test_dir = os.path.join(split_base_dir, "test")

IMG_SIZE = (224, 224)

# ImageDataGenerators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=20, class_mode='categorical', target_size=IMG_SIZE, color_mode='grayscale', shuffle=True
)
validation_generator = val_test_datagen.flow_from_directory(
    val_dir, batch_size=20, class_mode='categorical', target_size=IMG_SIZE, color_mode='grayscale'
)
test_generator = val_test_datagen.flow_from_directory(
    test_dir, batch_size=20, class_mode='categorical', target_size=IMG_SIZE, color_mode='grayscale', shuffle=False
)

# MODEL: 1D-CNN + GRU
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((224, 224), input_shape=(224, 224, 1)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.GRU(128, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# TRAIN
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

model.save("cnn_gru_model.h5")

# EVALUATE
start_time = time.time()
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc:.4f}")

# PREDICTIONS
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = list(test_generator.class_indices.keys())

# CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# PLOT CONFUSION MATRIX
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# ADDITIONAL METRICS
print("\nAdditional Metrics:")
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred, average='weighted'), 4))
print("Recall:", round(recall_score(y_true, y_pred, average='weighted'), 4))
print("F1 Score:", round(f1_score(y_true, y_pred, average='weighted'), 4))
print("Total Time (seconds):", round(time.time() - start_time, 2))

