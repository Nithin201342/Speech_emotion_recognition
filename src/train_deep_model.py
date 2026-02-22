"""
train_deep_model.py - Train a simple Neural Network for Speech Emotion Recognition

Architecture:
    Input (182 features)
      → Dense(256, relu)
      → Dropout(0.3)
      → Dense(128, relu)
      → Dropout(0.3)
      → Dense(8, softmax)   ← 8 emotion classes

Steps:
  1. Load processed .npy files
  2. Convert labels to one-hot vectors (required for categorical crossentropy)
  3. Build the neural network
  4. Train for 30 epochs with a validation split
  5. Print final test accuracy
  6. Compare with Random Forest accuracy from Day 3
  7. Plot accuracy and loss curves
  8. Save the model as models/deep_model.h5
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow info messages — only show warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------
PROCESSED_DIR   = "data/processed"
MODEL_SAVE_PATH = "models/deep_model.h5"
OUTPUT_DIR      = "outputs"

# From Day 3 — used for comparison at the end
RANDOM_FOREST_ACCURACY = 49.31  # percent


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition — Neural Network")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # Step 1: Load the processed data
    # Same files that were used to train the Random Forest model
    # -----------------------------------------------------------------------
    print("\n[1] Loading processed data...")

    X_train       = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test        = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train       = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test        = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    label_classes = np.load(os.path.join(PROCESSED_DIR, "label_classes.npy"), allow_pickle=True)

    num_classes  = len(label_classes)  # 8 emotions
    input_size   = X_train.shape[1]   # 182 features

    print(f"    Training samples : {X_train.shape[0]}")
    print(f"    Testing samples  : {X_test.shape[0]}")
    print(f"    Feature size     : {input_size}")
    print(f"    Number of classes: {num_classes}")

    # -----------------------------------------------------------------------
    # Step 2: Convert labels to one-hot encoding
    #
    # What is one-hot encoding?
    # The label "happy" maps to index 4.
    # One-hot turns that index into a vector:  [0, 0, 0, 0, 1, 0, 0, 0]
    # The neural network outputs one probability per class,
    # so labels must also be in this format for loss calculation.
    # -----------------------------------------------------------------------
    print("\n[2] Converting labels to one-hot format...")
    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes=num_classes)

    # -----------------------------------------------------------------------
    # Step 3: Build the neural network
    #
    # Architecture (simple, 3 steps):
    #   Dense(256)  → learns patterns in the 182 audio features
    #   Dropout     → randomly switches off 30% of neurons to avoid overfitting
    #   Dense(128)  → learns more abstract patterns
    #   Dropout     → another dropout for regularization
    #   Dense(8)    → outputs probability for each of the 8 emotions
    #
    # Activation functions:
    #   relu   → keeps positive values, sets negatives to 0 (default choice)
    #   softmax → converts outputs to probabilities that sum to 1
    # -----------------------------------------------------------------------
    print("\n[3] Building neural network...")

    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_size,)),
        Dropout(0.3),    # drop 30% of neurons randomly during training
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),  # output layer
    ])

    # Print model structure
    model.summary()

    # -----------------------------------------------------------------------
    # Step 4: Compile the model
    #
    # optimizer = adam  → a popular optimizer that adjusts learning rate auto
    # loss = categorical_crossentropy  → standard loss for multi-class problems
    # metrics = accuracy  → we want to track accuracy during training
    # -----------------------------------------------------------------------
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # -----------------------------------------------------------------------
    # Step 5: Train the model
    #
    # epochs = 30   → we go through the training data 30 times
    # batch_size = 32  → update weights after every 32 samples
    # validation_split = 0.1  → use 10% of training data to check overfitting
    #
    # EarlyStopping: if validation accuracy stops improving for 5 epochs,
    # stop training early and restore the best weights
    # -----------------------------------------------------------------------
    print("\n[4] Training the model (up to 30 epochs)...")

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,              # wait 5 epochs before stopping
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train_oh,
        epochs=30,
        batch_size=32,
        validation_split=0.1,   # 10% of training data for validation
        callbacks=[early_stop],
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # Step 6: Evaluate on test set
    # -----------------------------------------------------------------------
    print("\n[5] Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_oh, verbose=0)
    test_accuracy_pct = test_accuracy * 100

    print(f"\n  Neural Network Test Accuracy: {test_accuracy_pct:.2f}%")

    # -----------------------------------------------------------------------
    # Step 7: Compare with Random Forest from Day 3
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"  Random Forest (Day 3) : {RANDOM_FOREST_ACCURACY:.2f}%")
    print(f"  Neural Network (Day 4): {test_accuracy_pct:.2f}%")

    if test_accuracy_pct > RANDOM_FOREST_ACCURACY:
        diff = test_accuracy_pct - RANDOM_FOREST_ACCURACY
        print(f"\n  ✅ Neural Network is BETTER by {diff:.2f}%")
    else:
        diff = RANDOM_FOREST_ACCURACY - test_accuracy_pct
        print(f"\n  ℹ️  Random Forest is better by {diff:.2f}%")
        print("     (This can happen — neural networks need more data to shine)")

    print("=" * 55)

    # -----------------------------------------------------------------------
    # Step 8: Plot training curves
    # Accuracy vs Epoch and Loss vs Epoch saved to outputs/
    # -----------------------------------------------------------------------
    print("\n[6] Saving training curve plots...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Accuracy ---
    ax1.plot(epochs_range, history.history["accuracy"],     label="Train Accuracy", color="#1565C0")
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy",   color="#EF5350", linestyle="--")
    ax1.set_title("Accuracy vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Loss ---
    ax2.plot(epochs_range, history.history["loss"],     label="Train Loss", color="#1565C0")
    ax2.plot(epochs_range, history.history["val_loss"], label="Val Loss",   color="#EF5350", linestyle="--")
    ax2.set_title("Loss vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Neural Network Training History", fontsize=13)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    Saved: {plot_path}")

    # -----------------------------------------------------------------------
    # Step 9: Save the trained model
    # -----------------------------------------------------------------------
    print("\n[7] Saving neural network model...")
    model.save(MODEL_SAVE_PATH)
    print(f"    Model saved to: {MODEL_SAVE_PATH}")

    print("\n  Done! Check outputs/training_curves.png to see how training went.")
    print("=" * 55)


if __name__ == "__main__":
    main()
