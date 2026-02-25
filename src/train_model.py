# Trains a Random Forest classifier on the prepared audio features.

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PROCESSED_DIR = "data/processed"
MODEL_PATH     = "models/model.pkl"


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition - Model Training")
    print("=" * 55)

    # Load the preprocessed training and test data
    print("\n[1] Loading preprocessed data...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Load emotion names for use in the report
    label_classes = np.load(
        os.path.join(PROCESSED_DIR, "label_classes.npy"), allow_pickle=True
    )

    print(f"    Training samples : {X_train.shape[0]}")
    print(f"    Testing samples  : {X_test.shape[0]}")
    print(f"    Feature size     : {X_train.shape[1]}")

    # Train a Random Forest with 100 trees using all CPU cores
    print("\n[2] Training Random Forest model...")
    print("    (This may take a moment)")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("    Model training complete!")

    # Predict on the test set
    print("\n[3] Predicting on test data...")
    y_pred = model.predict(X_test)

    # Print accuracy, confusion matrix, and per-emotion report
    print("\n[4] Evaluating model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Model Accuracy: {accuracy * 100:.2f}%")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_classes))

    # Save the trained model to disk
    print("[5] Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"    Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Final Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
