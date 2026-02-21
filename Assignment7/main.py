import os
import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# -------------------- Configuration --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 64
IMG_SIZE = (224, 224)
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-4
SUPERVISED_EPOCHS = 10        # increased for better learning
SEMI_EPOCHS = 5
PSEUDO_LABEL_THRESHOLD = 0.95
LABELED_RATIO = 0.1
ITERATIONS = 2                 # two rounds to show iterative improvement

# Reduce sample sizes for faster processing
MAX_LABELED = 1000
MAX_UNLABELED = 2000
MAX_TEST = 1000

# -------------------- Data Augmentation --------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
# Note: Random flips and rotations are safe for CIFAR‑10. Other augmentations
# (e.g., brightness, contrast) were tested but did not improve performance.

# -------------------- Data Loading and Preprocessing --------------------
def load_and_preprocess_cifar10():
    print("=" * 80)
    print("LOADING CIFAR-10 DATASET")
    print("=" * 80)

    dataset = load_dataset('cifar10')
    train_data = dataset['train']
    test_data = dataset['test']

    train_indices = list(range(len(train_data)))
    random.shuffle(train_indices)
    n_labeled = min(int(len(train_data) * LABELED_RATIO), MAX_LABELED)
    n_unlabeled = min(len(train_data) - n_labeled, MAX_UNLABELED)
    labeled_indices = train_indices[:n_labeled]
    unlabeled_indices = train_indices[n_labeled:n_labeled + n_unlabeled]

    print(f"Total training samples: {len(train_data)}")
    print(f"Labeled samples: {len(labeled_indices)} (max {MAX_LABELED})")
    print(f"Unlabeled samples: {len(unlabeled_indices)} (max {MAX_UNLABELED})")
    print(f"Test samples: {len(test_data)} (max {MAX_TEST})")

    def load_images(indices, source):
        images, labels = [], []
        for i in indices:
            img = source[i]['img']
            img = Image.fromarray(np.array(img)).resize(IMG_SIZE)
            images.append(np.array(img, dtype=np.float32))
            labels.append(source[i]['label'])
        return np.array(images), np.array(labels)

    X_labeled, y_labeled = load_images(labeled_indices, train_data)
    X_unlabeled, _ = load_images(unlabeled_indices, train_data)
    test_indices = list(range(len(test_data)))[:MAX_TEST]
    X_test, y_test = load_images(test_indices, test_data)

    # Normalize pixel values to [0,1] (we'll apply ResNet preprocessing later)
    X_labeled /= 255.0
    X_unlabeled /= 255.0
    X_test /= 255.0

    print(f"\nLabeled shape: {X_labeled.shape}")
    print(f"Unlabeled shape: {X_unlabeled.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_labeled, y_labeled, X_unlabeled, X_test, y_test

# -------------------- ResNet50 with proper preprocessing --------------------
def build_resnet50_model(num_classes=10):
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

# -------------------- Pseudo-Label Generation --------------------
def generate_pseudo_labels(model, X_unlabeled, threshold=PSEUDO_LABEL_THRESHOLD, batch_size=BATCH_SIZE):
    print(f"\nGenerating pseudo-labels with threshold {threshold}...")
    probs = model.predict(X_unlabeled, batch_size=batch_size, verbose=1)
    pseudo_labels = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    mask = confidence > threshold
    X_pseudo = X_unlabeled[mask]
    y_pseudo = pseudo_labels[mask]

    print(f"Retained {len(X_pseudo)} / {len(X_unlabeled)} pseudo-labeled samples ({len(X_pseudo)/len(X_unlabeled)*100:.1f}%)")
    return X_pseudo, y_pseudo

# -------------------- Training --------------------
def train_supervised(model, train_ds, val_ds, epochs, lr=INITIAL_LR):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
    return history

def fine_tune_with_pseudo(model, base_model, combined_ds, val_ds, epochs, lr=FINE_TUNE_LR):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(combined_ds, epochs=epochs, validation_data=val_ds, verbose=1)
    return history

# -------------------- Evaluation --------------------
def evaluate_and_visualize(model, test_ds, y_test, histories, labels, output_dir='.'):
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Semi-Supervised ResNet50')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    print("Saved: confusion_matrix.png")

    plt.figure(figsize=(14, 5))
    for i, (history, label) in enumerate(zip(histories, labels)):
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], linestyle='-', label=f'{label} Train')
        plt.plot(history.history['val_accuracy'], linestyle='--', label=f'{label} Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], linestyle='-', label=f'{label} Train')
        plt.plot(history.history['val_loss'], linestyle='--', label=f'{label} Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle('ResNet50 Semi-Supervised Learning on CIFAR-10', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    print("Saved: training_curves.png")

# -------------------- Main --------------------
def main():
    os.makedirs('Assignment7', exist_ok=True)

    X_labeled, y_labeled, X_unlabeled, X_test, y_test = load_and_preprocess_cifar10()

    # Create datasets with augmentation for training
    labeled_ds = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled))
    labeled_ds = labeled_ds.shuffle(10000).batch(BATCH_SIZE)
    # Apply augmentation to the labeled training batches
    labeled_ds = labeled_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    labeled_ds = labeled_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build model
    model, base_model = build_resnet50_model()
    print("\n" + "=" * 80)
    print("INITIAL SUPERVISED TRAINING (10% labeled data)")
    print("=" * 80)
    history_sup = train_supervised(model, labeled_ds, test_ds, epochs=SUPERVISED_EPOCHS, lr=INITIAL_LR)

    histories = [history_sup]
    labels = ['Supervised']

    X_unlabeled_current = X_unlabeled.copy()
    for iteration in range(1, ITERATIONS + 1):
        print("\n" + "=" * 80)
        print(f"PSEUDO-LABELING ITERATION {iteration}")
        print("=" * 80)

        X_pseudo, y_pseudo = generate_pseudo_labels(model, X_unlabeled_current)

        if len(X_pseudo) == 0:
            print("No high-confidence pseudo-labels obtained. Stopping.")
            break

        X_combined = np.concatenate([X_labeled, X_pseudo], axis=0)
        y_combined = np.concatenate([y_labeled, y_pseudo], axis=0)
        combined_ds = tf.data.Dataset.from_tensor_slices((X_combined, y_combined))
        combined_ds = combined_ds.shuffle(20000).batch(BATCH_SIZE)
        # Apply augmentation to the combined training set as well
        combined_ds = combined_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        combined_ds = combined_ds.prefetch(tf.data.AUTOTUNE)

        history_ft = fine_tune_with_pseudo(model, base_model, combined_ds, test_ds, epochs=SEMI_EPOCHS, lr=FINE_TUNE_LR)
        histories.append(history_ft)
        labels.append(f'Iteration {iteration}')

    evaluate_and_visualize(model, test_ds, y_test, histories, labels, output_dir='Assignment7')
    model.save('Assignment7/resnet50_semi_final.keras')
    print("\nModel saved: Assignment7/resnet50_semi_final.keras")

if __name__ == "__main__":
    main()