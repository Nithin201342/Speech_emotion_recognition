"""
evaluate_model.py - Evaluate the trained model and save result charts

This script:
  1. Loads the test data (X_test.npy, y_test.npy)
  2. Loads the trained model (models/model.pkl)
  3. Predicts on the test set
  4. Saves a confusion matrix heatmap → outputs/confusion_matrix.png
  5. Saves a per-emotion accuracy bar chart → outputs/accuracy_per_emotion.png
  6. Prints the classification report to the terminal
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# -----------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------
PROCESSED_DIR = "data/processed"
MODEL_PATH    = "models/model.pkl"
OUTPUT_DIR    = "outputs"


def plot_confusion_matrix(cm, labels, save_path):
    """
    Plot and save a confusion matrix as a heatmap.
    Rows = actual emotion, Columns = predicted emotion.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,         # show numbers inside each cell
        fmt="d",            # show as integers
        cmap="Blues",       # colour scheme
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix — Speech Emotion Recognition", fontsize=14)
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.ylabel("Actual Emotion", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_accuracy_per_emotion(y_test, y_pred, labels, save_path):
    """
    Plot and save a bar chart of accuracy for each individual emotion.
    This makes it easy to see which emotions the model is best/worst at.
    """
    # Calculate per-emotion accuracy
    per_emotion_accuracy = []

    for i, emotion in enumerate(labels):
        # Find all test samples that are actually this emotion
        indices = np.where(y_test == i)[0]

        if len(indices) == 0:
            per_emotion_accuracy.append(0.0)
            continue

        # How many of those did the model get right?
        correct = np.sum(y_pred[indices] == i)
        accuracy = correct / len(indices) * 100
        per_emotion_accuracy.append(accuracy)

    # Pick bar colours — green for high accuracy, red for low
    colors = ["#4CAF50" if acc >= 70 else "#FF5252" for acc in per_emotion_accuracy]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, per_emotion_accuracy, color=colors, edgecolor="black", linewidth=0.5)

    # Add the percentage value on top of each bar
    for bar, acc in zip(bars, per_emotion_accuracy):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Accuracy per Emotion — Random Forest", fontsize=14)
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition — Model Evaluation")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # Step 1: Load test data and model
    # -----------------------------------------------------------------------
    print("\n[1] Loading test data and model...")

    X_test        = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test        = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    label_classes = np.load(os.path.join(PROCESSED_DIR, "label_classes.npy"), allow_pickle=True)
    model         = joblib.load(MODEL_PATH)

    print(f"    Test samples: {X_test.shape[0]}")

    # -----------------------------------------------------------------------
    # Step 2: Predict on test set
    # -----------------------------------------------------------------------
    print("\n[2] Running predictions on test set...")
    y_pred = model.predict(X_test)

    # -----------------------------------------------------------------------
    # Step 3: Print classification report
    # -----------------------------------------------------------------------
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_classes))

    # -----------------------------------------------------------------------
    # Step 4: Save charts
    # -----------------------------------------------------------------------
    print("[3] Saving evaluation charts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        label_classes,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )

    plot_accuracy_per_emotion(
        y_test,
        y_pred,
        label_classes,
        os.path.join(OUTPUT_DIR, "accuracy_per_emotion.png"),
    )

    print("\n  All charts saved to the 'outputs/' folder.")
    print("=" * 55)


if __name__ == "__main__":
    main()
