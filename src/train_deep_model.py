# Trains a simple neural network for speech emotion recognition using Keras.

import os
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

PROCESSED_DIR         = "data/processed"
MODEL_SAVE_PATH       = "models/deep_model.h5"
OUTPUT_DIR            = "outputs"
RANDOM_FOREST_ACCURACY = 49.31  # kept for comparison at the end


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition — Neural Network")
    print("=" * 55)

    # Load the same processed data used for the Random Forest model
    print("\n[1] Loading processed data...")
    X_train       = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test        = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train       = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test        = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    label_classes = np.load(os.path.join(PROCESSED_DIR, "label_classes.npy"), allow_pickle=True)

    num_classes = len(label_classes)
    input_size  = X_train.shape[1]

    print(f"    Training samples : {X_train.shape[0]}")
    print(f"    Testing samples  : {X_test.shape[0]}")
    print(f"    Feature size     : {input_size}")
    print(f"    Number of classes: {num_classes}")

    # Convert integer labels to one-hot vectors required by categorical crossentropy
    print("\n[2] Converting labels to one-hot format...")
    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes=num_classes)

    # Build a 3-layer dense network with dropout for regularisation
    print("\n[3] Building neural network...")
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_size,)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.summary()

    # Compile with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train for up to 30 epochs, stopping early if validation stops improving
    print("\n[4] Training the model (up to 30 epochs)...")
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train_oh,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate on the test set
    print("\n[5] Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_oh, verbose=0)
    test_accuracy_pct = test_accuracy * 100
    print(f"\n  Neural Network Test Accuracy: {test_accuracy_pct:.2f}%")

    # Compare neural network accuracy against Random Forest
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"  Random Forest  : {RANDOM_FOREST_ACCURACY:.2f}%")
    print(f"  Neural Network : {test_accuracy_pct:.2f}%")

    if test_accuracy_pct > RANDOM_FOREST_ACCURACY:
        diff = test_accuracy_pct - RANDOM_FOREST_ACCURACY
        print(f"\n  Neural Network is better by {diff:.2f}%")
    else:
        diff = RANDOM_FOREST_ACCURACY - test_accuracy_pct
        print(f"\n  Random Forest is better by {diff:.2f}%")
        print("     (Neural networks generally need more data to outperform.)")
    print("=" * 55)

    # Save accuracy and loss curves to outputs/
    print("\n[6] Saving training curve plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax1.plot(epochs_range, history.history["accuracy"],     label="Train Accuracy", color="#1565C0")
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy",   color="#EF5350", linestyle="--")
    ax1.set_title("Accuracy vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
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

    # Save the trained neural network model
    print("\n[7] Saving neural network model...")
    model.save(MODEL_SAVE_PATH)
    print(f"    Model saved to: {MODEL_SAVE_PATH}")

    print("\n  Done! Check outputs/training_curves.png to see how training went.")
    print("=" * 55)


if __name__ == "__main__":
    main()
