import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import cv2
import matplotlib.pyplot as plt
import json

# ============================================================================
# FASTER CONFIGURATION
# ============================================================================
IMG_SIZE = 128  # REDUCED from 224 for SPEED
BATCH_SIZE = 16  # INCREASED for SPEED
EPOCHS = 30  # REDUCED from 100 for SPEED
K_FOLDS = 3  # REDUCED from 5 for SPEED
AUGMENT_FACTOR = 3  # REDUCED from 5 for SPEED
DATASET_PATH = 'kvasir-dataset'

CLASS_NAMES = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

RISK_MAPPING = {
    'dyed-lifted-polyps': 'High Risk',
    'dyed-resection-margins': 'High Risk',
    'esophagitis': 'Medium Risk',
    'normal-cecum': 'Low Risk',
    'normal-pylorus': 'Low Risk',
    'normal-z-line': 'Low Risk',
    'polyps': 'High Risk',
    'ulcerative-colitis': 'High Risk'
}

print("="*80)
print("⚡ FAST K-FOLD TRAINING - OPTIMIZED FOR SPEED")
print("="*80)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} (smaller = faster)")
print(f"Batch Size: {BATCH_SIZE} (larger = faster)")
print(f"K-Folds: {K_FOLDS} (fewer = faster)")
print(f"Epochs: {EPOCHS} (fewer = faster)")
print(f"Augmentation: {AUGMENT_FACTOR}x (less = faster)")
print("="*80 + "\n")

# Enable GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✓ GPU detected and configured\n")

# ============================================================================
# LOAD IMAGES
# ============================================================================
def load_all_images(dataset_path):
    images = []
    labels = []
    
    print("📁 Loading images...")
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            img_path = os.path.join(class_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(class_idx)
                
            except:
                continue
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

X_all, y_all = load_all_images(DATASET_PATH)
print(f"✓ Loaded {len(X_all)} images ({X_all.shape})\n")

# ============================================================================
# FAST AUGMENTATION
# ============================================================================
def augment_batch(images):
    """Vectorized augmentation for speed"""
    batch = images.copy()
    
    # Random flips (vectorized)
    flip_h = np.random.random(len(batch)) > 0.5
    flip_v = np.random.random(len(batch)) > 0.5
    
    for i in range(len(batch)):
        if flip_h[i]:
            batch[i] = np.fliplr(batch[i])
        if flip_v[i]:
            batch[i] = np.flipud(batch[i])
    
    return batch

def create_augmented_dataset(X, y, factor=3):
    """Fast augmentation"""
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(factor - 1):
        X_aug.append(augment_batch(X))
        y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)

# ============================================================================
# BUILD LIGHTWEIGHT MODEL
# ============================================================================
def build_model():
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=0.5  # Use smaller MobileNet for speed
    )
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Less dropout for speed
    x = Dense(32, activation='relu')(x)  # Smaller layer
    outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

# ============================================================================
# K-FOLD CROSS VALIDATION
# ============================================================================
print(f"🔄 {K_FOLDS}-FOLD CROSS-VALIDATION\n")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

fold_accuracies = []
all_predictions = []
all_true_labels = []

fold_num = 1

for train_index, val_index in skf.split(X_all, y_all):
    print(f"{'='*80}")
    print(f"📂 FOLD {fold_num}/{K_FOLDS}")
    print(f"{'='*80}")
    
    X_train_fold = X_all[train_index]
    y_train_fold = y_all[train_index]
    X_val_fold = X_all[val_index]
    y_val_fold = y_all[val_index]
    
    # Fast augmentation
    print(f"🔄 Augmenting ({AUGMENT_FACTOR}x)...", end=" ")
    X_train_aug, y_train_aug = create_augmented_dataset(X_train_fold, y_train_fold, AUGMENT_FACTOR)
    print(f"✓ {len(X_train_aug)} images")
    
    y_train_cat = tf.keras.utils.to_categorical(y_train_aug, num_classes=len(CLASS_NAMES))
    y_val_cat = tf.keras.utils.to_categorical(y_val_fold, num_classes=len(CLASS_NAMES))
    
    # Build model
    model, base_model = build_model()
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Higher LR for speed
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Use NATIVE KERAS FORMAT (.keras)
    checkpoint = ModelCheckpoint(
        f'fold_{fold_num}_best.keras',  # Changed to .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    # Train
    print(f"🚀 Training ({EPOCHS} epochs)...")
    history = model.fit(
        X_train_aug, y_train_cat,
        validation_data=(X_val_fold, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
        verbose=0
    )
    
    # Load best
    model = tf.keras.models.load_model(f'fold_{fold_num}_best.keras')
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_cat, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_val_fold, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    all_predictions.extend(y_pred_classes)
    all_true_labels.extend(y_val_fold)
    fold_accuracies.append(val_acc)
    
    print(f"✓ Accuracy: {val_acc*100:.2f}%\n")
    
    fold_num += 1

# ============================================================================
# RESULTS
# ============================================================================
print("="*80)
print("🎯 FINAL RESULTS")
print("="*80)

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"\n📈 K-Fold Results:")
print(f"   Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"   Individual Folds: {[f'{acc*100:.1f}%' for acc in fold_accuracies]}")

if mean_acc >= 0.75:
    print("\n✅ GOOD for 192 images!")
elif mean_acc >= 0.60:
    print("\n✓ Acceptable")
else:
    print("\n⚠️  Low accuracy")

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)

print("\n📋 Classification Report:")
print(classification_report(
    all_true_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=3,
    zero_division=0
))

print("\n📈 Per-Class Accuracy:")
for i, class_name in enumerate(CLASS_NAMES):
    mask = np.array(all_true_labels) == i
    if mask.sum() > 0:
        correct = ((np.array(all_predictions) == i) & mask).sum()
        total = mask.sum()
        acc = correct / total
        print(f"   {class_name:30s}: {acc*100:5.1f}% ({correct}/{total})")

# ============================================================================
# TRAIN FINAL MODEL
# ============================================================================
print("\n" + "="*80)
print("🏗️  TRAINING FINAL MODEL")
print("="*80)

print(f"🔄 Augmenting full dataset ({AUGMENT_FACTOR}x)...", end=" ")
X_all_aug, y_all_aug = create_augmented_dataset(X_all, y_all, AUGMENT_FACTOR)
print(f"✓ {len(X_all_aug)} images")

y_all_cat = tf.keras.utils.to_categorical(y_all_aug, num_classes=len(CLASS_NAMES))

final_model, final_base = build_model()

final_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

checkpoint_final = ModelCheckpoint(
    'final_model.keras',  # Native format
    monitor='loss',
    save_best_only=True,
    verbose=0
)

print(f"🚀 Training ({EPOCHS} epochs)...")
history_final = final_model.fit(
    X_all_aug, y_all_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_final],
    verbose=1
)

# Load best
final_model = tf.keras.models.load_model('final_model.keras')

# ============================================================================
# FINE-TUNING (FAST)
# ============================================================================
print("\n🔧 Fine-tuning (10 epochs)...")

final_base.trainable = True
for layer in final_base.layers[:-10]:
    layer.trainable = False

final_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

checkpoint_ft = ModelCheckpoint(
    'final_model_finetuned.keras',
    monitor='loss',
    save_best_only=True,
    verbose=0
)

history_ft = final_model.fit(
    X_all_aug, y_all_cat,
    epochs=10,  # Just 10 epochs for fine-tuning
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_ft],
    verbose=1
)

final_model = tf.keras.models.load_model('final_model_finetuned.keras')

# ============================================================================
# SAVE
# ============================================================================
print("\n💾 Saving...")
final_model.save('gi_cancer_final.keras')  # Native format

metadata = {
    'classes': CLASS_NAMES,
    'risk_mapping': RISK_MAPPING,
    'k_folds': K_FOLDS,
    'mean_accuracy': float(mean_acc),
    'std_accuracy': float(std_acc),
    'img_size': IMG_SIZE,
    'total_images': int(len(X_all)),
    'model': 'MobileNetV2_Fast'
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Quick plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, K_FOLDS+1), [acc*100 for acc in fold_accuracies])
plt.axhline(mean_acc*100, color='r', linestyle='--', label=f'Mean: {mean_acc*100:.1f}%')
plt.ylabel('Accuracy (%)')
plt.xlabel('Fold')
plt.title('K-Fold Results')
plt.legend()
plt.ylim(0, 100)

plt.subplot(1, 2, 2)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('results.png', dpi=150)

print("\n✅ DONE!")
print("="*80)
print(f"Final Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print("\nFiles saved:")
print("  ✓ gi_cancer_final.keras")
print("  ✓ metadata.json")
print("  ✓ results.png")
print("="*80)
