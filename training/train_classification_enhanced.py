"""
Enhanced Classification Model Training Script
Generates ALL required outputs for IEEE-level project presentation:
- Confusion Matrix
- Accuracy/Loss Graphs
- Classification Report
- ROC Curve
- Training History
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = "dataset_classification"
OUTPUT_DIR = "training_outputs"
MODEL_NAME = "fracture_classification_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("FRACTURESENSE AI - CLASSIFICATION MODEL TRAINING")
print("=" * 60)
print(f"Dataset: {DATASET_PATH}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)

# ============================================
# 1. DATA LOADING & PREPROCESSING
# ============================================
print("\n[1/8] Loading and preprocessing data...")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# Load training data
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save class labels
class_indices = train_data.class_indices
class_names = list(class_indices.keys())
num_classes = len(class_names)

with open(os.path.join(OUTPUT_DIR, "classify_classes.json"), "w") as f:
    json.dump({"classes": class_names, "indices": class_indices}, f, indent=4)

print(f"✓ Training samples: {train_data.samples}")
print(f"✓ Validation samples: {val_data.samples}")
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Classes: {class_names}")

# ============================================
# 2. MODEL ARCHITECTURE
# ============================================
print("\n[2/8] Building model architecture...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("✓ Model compiled successfully")
print(f"✓ Total parameters: {model.count_params():,}")
model.summary()

# ============================================
# 3. TRAINING CALLBACKS
# ============================================
print("\n[3/8] Setting up training callbacks...")

callbacks = [
    ModelCheckpoint(
        MODEL_NAME,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("✓ Callbacks configured")

# ============================================
# 4. MODEL TRAINING
# ============================================
print("\n[4/8] Training model...")
print("This may take a while depending on your hardware...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("✓ Training completed!")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

# ============================================
# 5. GENERATE ACCURACY & LOSS GRAPHS
# ============================================
print("\n[5/8] Generating accuracy and loss graphs...")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Plot 2: Loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_loss_graphs.png'), dpi=300, bbox_inches='tight')
plt.close()

# Individual graphs
# Accuracy Graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
plt.fill_between(range(len(history.history['accuracy'])), 
                 history.history['accuracy'], alpha=0.3)
plt.fill_between(range(len(history.history['val_accuracy'])), 
                 history.history['val_accuracy'], alpha=0.3)
plt.title('Classification Model - Accuracy', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_graph.png'), dpi=300, bbox_inches='tight')
plt.close()

# Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
plt.fill_between(range(len(history.history['loss'])), 
                 history.history['loss'], alpha=0.3)
plt.fill_between(range(len(history.history['val_loss'])), 
                 history.history['val_loss'], alpha=0.3)
plt.title('Classification Model - Loss', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_graph.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Accuracy and loss graphs saved")

# ============================================
# 6. GENERATE PREDICTIONS FOR VALIDATION SET
# ============================================
print("\n[6/8] Generating predictions for validation set...")

# Get all validation predictions
val_data.reset()
y_true = val_data.classes
y_pred_proba = model.predict(val_data, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"✓ Generated {len(y_pred)} predictions")

# ============================================
# 7. CONFUSION MATRIX
# ============================================
print("\n[7/8] Generating confusion matrix...")

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Number of Samples'},
    square=True
)
plt.title('Confusion Matrix - Fracture Classification', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Confusion matrix saved")

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_normalized, 
    annot=True, 
    fmt='.2%', 
    cmap='RdYlGn',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Percentage'},
    square=True,
    vmin=0,
    vmax=1
)
plt.title('Normalized Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Normalized confusion matrix saved")

# ============================================
# 8. CLASSIFICATION REPORT & ROC CURVE
# ============================================
print("\n[8/8] Generating classification report and ROC curves...")

# Classification Report
report = classification_report(
    y_true, 
    y_pred, 
    target_names=class_names,
    digits=4
)

# Save classification report
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("FRACTURESENSE AI - CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {DATASET_PATH}\n")
    f.write(f"Total Samples: {len(y_true)}\n")
    f.write(f"Number of Classes: {num_classes}\n\n")
    f.write("=" * 60 + "\n")
    f.write("CLASSIFICATION METRICS\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")

print("✓ Classification report saved")

# ROC Curve (One-vs-Rest)
if num_classes > 2:
    # Binarize labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i], tpr[i], 
            color=color, 
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ ROC curves saved")

# ============================================
# GENERATE SUMMARY REPORT
# ============================================
print("\nGenerating performance summary...")

# Calculate final metrics
final_metrics = {
    'Training Accuracy': history.history['accuracy'][-1],
    'Validation Accuracy': history.history['val_accuracy'][-1],
    'Training Loss': history.history['loss'][-1],
    'Validation Loss': history.history['val_loss'][-1],
    'Total Epochs': len(history.history['accuracy']),
    'Best Validation Accuracy': max(history.history['val_accuracy']),
    'Total Parameters': model.count_params(),
}

# Save summary
with open(os.path.join(OUTPUT_DIR, 'training_summary.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("FRACTURESENSE AI - TRAINING SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model Name: {MODEL_NAME}\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset Path: {DATASET_PATH}\n\n")
    
    f.write("FINAL METRICS:\n")
    f.write("-" * 60 + "\n")
    for key, value in final_metrics.items():
        if 'Accuracy' in key or 'Loss' in key:
            f.write(f"{key:30s}: {value:.4f}\n")
        else:
            f.write(f"{key:30s}: {value:,}\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("GENERATED OUTPUT FILES:\n")
    f.write("=" * 60 + "\n")
    f.write("✓ accuracy_loss_graphs.png      - Combined accuracy/loss plot\n")
    f.write("✓ accuracy_graph.png            - Detailed accuracy plot\n")
    f.write("✓ loss_graph.png                - Detailed loss plot\n")
    f.write("✓ confusion_matrix.png          - Confusion matrix heatmap\n")
    f.write("✓ confusion_matrix_normalized.png - Normalized confusion matrix\n")
    f.write("✓ roc_curves.png                - ROC curves (if multi-class)\n")
    f.write("✓ classification_report.txt     - Detailed metrics report\n")
    f.write("✓ training_history.csv          - Epoch-by-epoch history\n")
    f.write("✓ training_summary.txt          - This file\n")
    f.write("✓ classify_classes.json         - Class labels mapping\n")
    f.write("\n" + "=" * 60 + "\n")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n✓ Model saved: {MODEL_NAME}")
print(f"✓ All outputs saved in: {OUTPUT_DIR}/")
print(f"✓ Final validation accuracy: {final_metrics['Validation Accuracy']:.2%}")
print(f"✓ Best validation accuracy: {final_metrics['Best Validation Accuracy']:.2%}")
print("\nGenerated Files:")
print("-" * 60)
for filename in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, filename)
    size = os.path.getsize(filepath) / 1024  # KB
    print(f"  • {filename:35s} ({size:.1f} KB)")

print("\n" + "=" * 60)
print("Ready for IEEE presentation! 🎉")
print("=" * 60)