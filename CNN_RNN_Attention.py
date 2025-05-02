import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Layer, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import random

# Paths and settings
dataset_path = ""
sequence_length = 30
num_classes = 3

# Data Loading
def load_dataset():
    X, y = [], []
    frame_indices_list = []
    for seizure_type in os.listdir(dataset_path):
        seizure_folder = os.path.join(dataset_path, seizure_type)
        if not os.path.isdir(seizure_folder):
            continue
        for patient in os.listdir(seizure_folder):
            patient_folder = os.path.join(seizure_folder, patient)
            if not os.path.isdir(patient_folder):
                continue
            frames = sorted([f for f in os.listdir(patient_folder) if f.endswith(('.png', '.jpg'))])
            num_frames = len(frames)
            for start_idx in range(0, num_frames - sequence_length + 1, sequence_length // 2):
                frame_sequence = []
                frame_indices = []
                for frame_file in frames[start_idx:start_idx+sequence_length]:
                    frame_path = os.path.join(patient_folder, frame_file)
                    frame = cv2.imread(frame_path)
                    frame = cv2.resize(frame, (224, 224))
                    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
                    frame_sequence.append(frame)
                    frame_indices.append(os.path.join(patient_folder, frame_file))
                if len(frame_sequence) == sequence_length:
                    X.append(frame_sequence)
                    y.append(seizure_type)
                    frame_indices_list.append(frame_indices)
    X = np.array(X)
    y = np.array(y)
    unique_labels = sorted(set(y))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    y = to_categorical(y, num_classes=len(unique_labels))
    return X, y, label_map, frame_indices_list

# Load Data
X, y, label_map, frame_indices_list = load_dataset()
print(f"Dataset Loaded: {X.shape}, Labels: {y.shape}, Label Map: {label_map}")

# Stratified split
y_labels = np.argmax(y, axis=1)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, frame_indices_list, test_size=0.15, random_state=42, stratify=y_labels)
y_train_labels = np.argmax(y_train, axis=1)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_train, y_train, idx_train, test_size=0.1765, random_state=42, stratify=y_train_labels)

# Feature Extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Sequential([
    base_model,
    GlobalAveragePooling2D()
])

def extract_features(X, batch_size=32):
    num_samples, seq_len, h, w, c = X.shape
    X_flat = X.reshape(-1, h, w, c)
    features = feature_extractor.predict(X_flat, batch_size=batch_size, verbose=1)
    return features.reshape(num_samples, seq_len, -1)

X_train_features = extract_features(X_train)
X_val_features = extract_features(X_val)
X_test_features = extract_features(X_test)

# Corrected Temporal Attention Layer
class TemporalAttention(Layer):
    def __init__(self, units=64, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.V = None

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='W')
        self.V = self.add_weight(shape=(self.units, 1),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='V')
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        score = K.dot(K.tanh(K.dot(inputs, self.W)), self.V)
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# Build Model
input_layer = Input(shape=(sequence_length, X_train_features.shape[-1]))
x = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = TemporalAttention(units=128)(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
output_layer = Dense(num_classes, activation='softmax')(x)

attention_model = Model(inputs=input_layer, outputs=output_layer)

# Compile Model
attention_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.Precision(name='precision')]
)

# Training Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

# Train Model
history = attention_model.fit(
    X_train_features, y_train,
    validation_data=(X_val_features, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
y_pred_prob = attention_model.predict(X_test_features)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_map.keys(), digits=4))

if len(set(y_true)) > 1:
    auc_score = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    print(f"AUC Score: {auc_score:.4f}")

# Confusion Matrix
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.tight_layout()
plt.show()

# Test Metrics
test_metrics = attention_model.evaluate(X_test_features, y_test, verbose=0)
print("\nTest Metrics:")
print(f"Loss: {test_metrics[0]:.4f}")
print(f"Accuracy: {test_metrics[1]:.4f}")
print(f"AUC: {test_metrics[2]:.4f}")
print(f"Recall: {test_metrics[3]:.4f}")
print(f"Precision: {test_metrics[4]:.4f}")

# -----------------------------------------------------------
# New Visualization Part: Temporal Attention Plot with Indexed Frames
# Show one random sample per each class (all 3 classes)
# -----------------------------------------------------------
def visualize_temporal_attention(X_seq, y_true_label, y_pred_label, model):
    # Use the L2 norm of each frame's feature vector as an attention score proxy.
    sample_features = X_seq[0]  # shape: (sequence_length, feature_dim)
    scores = np.linalg.norm(sample_features, axis=1)
    inv_label_map = {v: k for k, v in label_map.items()}
    
    plt.figure(figsize=(10, 2))
    plt.plot(range(len(scores)), scores, marker='o', color='orange')
    plt.title(f"Temporal Attention | True: {inv_label_map[y_true_label]} | Pred: {inv_label_map[y_pred_label]}")
    plt.xlabel("Frame Index")
    plt.ylabel("Attention Score (L2 norm)")
    plt.grid(True)
    plt.show()

def show_sequence_frames(base_dir, class_name, frame_files, sequence_len=10):
    plt.figure(figsize=(15, 3))
    for i, f in enumerate(frame_files[:sequence_len]):
        img = cv2.imread(f)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
        plt.subplot(1, sequence_len, i+1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"F{i+1}")
    plt.suptitle(f"Frames from sample | Class: {class_name}", fontsize=14)
    plt.show()

# Loop through test samples and show one random sample per class
samples_shown = set()
inv_label_map = {v: k for k, v in label_map.items()}
random_indices = list(range(len(X_test_features)))
random.shuffle(random_indices)

for i in random_indices:
    true_cls = y_true[i]
    if true_cls in samples_shown:
        continue
    visualize_temporal_attention(X_test_features[i:i+1], y_true[i], y_pred[i], attention_model)
    show_sequence_frames(dataset_path, inv_label_map[y_true[i]], idx_test[i], sequence_len=10)
    samples_shown.add(true_cls)
    if len(samples_shown) == num_classes:
        break
