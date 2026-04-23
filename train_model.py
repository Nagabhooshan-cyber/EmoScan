import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join('data', 'fer2013.csv')
MODEL_DIR   = 'model'
MODEL_PATH  = os.path.join(MODEL_DIR, 'emotion_model.h5')
IMG_SIZE    = 48
NUM_CLASSES = 7
BATCH_SIZE  = 64
EPOCHS      = 100       # EarlyStopping will halt before this if needed
LR          = 1e-3

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Load & parse FER-2013 CSV ─────────────────────────────────────────────────
def load_fer2013(path):
    print(f"[*] Loading dataset from {path} …")
    df = pd.read_csv(path)
    print(f"    Total rows in CSV: {len(df)}")

    X, y, usages, skipped = [], [], [], 0

    for _, row in df.iterrows():
        try:
            pixels = np.array(str(row['pixels']).split(), dtype=np.float32)
            # Skip corrupt rows — FER-2013 has some rows with wrong pixel count
            if len(pixels) != IMG_SIZE * IMG_SIZE:
                skipped += 1
                continue
            X.append(pixels.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0)
            y.append(int(row['emotion']))
            # Safely read Usage, defaulting to 'Training' if missing/NaN
            usages.append(str(row.get('Usage', 'Training')))
        except Exception:
            skipped += 1
            continue

    print(f"    Loaded: {len(X)} | Skipped (corrupt): {skipped}")

    X       = np.array(X,      dtype=np.float32)
    y       = np.array(y,      dtype=np.int32)
    usages  = np.array(usages)

    usage_counts = {u: int((usages == u).sum())
                    for u in ['Training', 'PublicTest', 'PrivateTest']}
    print(f"    Usage counts: {usage_counts}")

    # ── Split strategy ────────────────────────────────────────────────────────
    # If the CSV has proper Usage labels, use them.
    # If PublicTest / PrivateTest are missing (some Kaggle versions omit them),
    # fall back to a clean 80 / 10 / 10 random split.
    has_val  = usage_counts.get('PublicTest',  0) > 0
    has_test = usage_counts.get('PrivateTest', 0) > 0

    if has_val and has_test:
        X_train = X[usages == 'Training']
        y_train = y[usages == 'Training']
        X_val   = X[usages == 'PublicTest']
        y_val   = y[usages == 'PublicTest']
        X_test  = X[usages == 'PrivateTest']
        y_test  = y[usages == 'PrivateTest']
    else:
        # Fallback: stratified 80 / 10 / 10 split
        print("    [!] Usage splits missing — using 80/10/10 random split")
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Deep CNN architecture ─────────────────────────────────────────────────────
def build_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Block 1 — 64 filters
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2 — 128 filters
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3 — 256 filters
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4 — 512 filters
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Fully-connected classifier
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    m = models.Model(inp, out, name='EmotionCNN')
    m.summary()
    return m


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(DATA_PATH)

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

    # Class weights to handle FER-2013 imbalance
    cw = compute_class_weight('balanced',
                              classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights = {i: float(cw[i]) for i in range(NUM_CLASSES)}
    print("[*] Class weights:", {EMOTIONS[i]: round(v, 2)
                                 for i, v in class_weights.items()})

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = build_model()

    # Cosine decay with warm restarts
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LR,
        first_decay_steps=10,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1),
    ]

    print(f"\n[*] Training for up to {EPOCHS} epochs …\n")
    history = model.fit(
        datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val_oh),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # Reload saved best weights before evaluating (avoids shape issues in Colab)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # If test split is empty (some CSV versions lack PrivateTest), use last 10%
    if len(X_test) == 0:
        print("[!] Test split empty — using last 10% of training data")
        split  = int(len(X_train) * 0.90)
        X_test = X_train[split:]
        y_test_oh = y_train_oh[split:]

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=1)
    print(f"\n[✓] Test accuracy : {test_acc * 100:.2f}%")
    print(f"[✓] Test loss     : {test_loss:.4f}")

    plot_history(history)
    print(f"\n[✓] Model saved to {MODEL_PATH}")


# ── Plot training curves ──────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[✓] Training curves saved to {out}")


if __name__ == '__main__':
    train()