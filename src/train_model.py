"""
train_model.py - Train a Random Forest model for Speech Emotion Recognition

Steps:
  1. Load the preprocessed feature files from data/processed/
  2. Train a Random Forest classifier on the training data
  3. Predict emotions on the test data
  4. Print accuracy, confusion matrix, and classification report
  5. Save the trained model to models/model.pkl
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------
PROCESSED_DIR = "data/processed"
MODEL_PATH     = "models/model.pkl"


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition - Model Training")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # Step 1: Load the preprocessed data
    # These files were created by prepare_data.py in Day 2
    # -----------------------------------------------------------------------
    print("\n[1] Loading preprocessed data...")

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Load the emotion names so we can show labels in the report
    label_classes = np.load(
        os.path.join(PROCESSED_DIR, "label_classes.npy"), allow_pickle=True
    )

    print(f"    Training samples : {X_train.shape[0]}")
    print(f"    Testing samples  : {X_test.shape[0]}")
    print(f"    Feature size     : {X_train.shape[1]}")

    # -----------------------------------------------------------------------
    # Step 2: Create and train the Random Forest model
    #
    # What is Random Forest?
    # - It builds many decision trees (n_estimators=100 means 100 trees)
    # - Each tree votes on what emotion it thinks the audio is
    # - The emotion with the most votes wins
    # - This voting makes it more accurate and stable than a single tree
    # -----------------------------------------------------------------------
    print("\n[2] Training Random Forest model...")
    print("    (This may take a moment)")

    model = RandomForestClassifier(
        n_estimators=100,    # number of decision trees to build
        random_state=42,     # so results are the same every time we run
        n_jobs=-1            # use all CPU cores to speed up training
    )

    model.fit(X_train, y_train)
    print("    Model training complete!")

    # -----------------------------------------------------------------------
    # Step 3: Predict on the test set
    # The model has never seen these files during training
    # -----------------------------------------------------------------------
    print("\n[3] Predicting on test data...")
    y_pred = model.predict(X_test)

    # -----------------------------------------------------------------------
    # Step 4: Evaluate the model
    #
    # What is Accuracy?
    # - It is the % of test samples the model guessed correctly
    # - Example: 78% means it got 78 out of 100 right
    # - This is the simplest and most commonly used metric
    #
    # What is a Confusion Matrix?
    # - A table showing how often each emotion was correctly classified
    # - Rows are the true emotion, columns are the predicted emotion
    # - Diagonal = correctly predicted; off-diagonal = mistakes
    # -----------------------------------------------------------------------
    print("\n[4] Evaluating model...")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Model Accuracy: {accuracy * 100:.2f}%")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_classes))

    # -----------------------------------------------------------------------
    # Step 5: Save the trained model to a file
    # joblib saves the model so we can load it later without retraining
    # -----------------------------------------------------------------------
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
