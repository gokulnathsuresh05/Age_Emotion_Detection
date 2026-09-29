import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE = r"F:\Age_Emotion_Detection_Dataset"
MODEL_DIR = r"E:\Project\Age_Emotion_Detection\models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train(name, category, train_csv, val_csv, test_csv, label_col):

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    feature_dir = os.path.join(BASE, "features", category)

    if category == "emotion":
        csv_dir = os.path.join(BASE, "emotion", "final_prepare")
    else:
        csv_dir = os.path.join(BASE, "final_prepare")

    X_train = np.load(os.path.join(feature_dir, train_csv.replace(".csv", "_features.npy")))
    X_val   = np.load(os.path.join(feature_dir, val_csv.replace(".csv", "_features.npy")))
    X_test  = np.load(os.path.join(feature_dir, test_csv.replace(".csv", "_features.npy")))

    train_df = pd.read_csv(os.path.join(csv_dir, train_csv))
    val_df   = pd.read_csv(os.path.join(csv_dir, val_csv))
    test_df  = pd.read_csv(os.path.join(csv_dir, test_csv))

    y_train = train_df[label_col].astype(str)
    y_val   = val_df[label_col].astype(str)
    y_test  = test_df[label_col].astype(str)

    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    print("\nStarting Random Forest training...")
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("\nValidation Accuracy:", round(accuracy_score(y_val, val_pred) * 100, 2), "%")
    print("Test Accuracy      :", round(accuracy_score(y_test, test_pred) * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    model_path = os.path.join(MODEL_DIR, name + ".joblib")
    joblib.dump(model, model_path)

    print("MODEL SAVED:", model_path)

train(
    "gender_model",
    "gender",
    "gender_train.csv",
    "gender_val.csv",
    "gender_test.csv",
    "gender_label"
)

train(
    "age_model",
    "age",
    "age_train.csv",
    "age_val.csv",
    "age_test.csv",
    "age_label"
)

train(
    "emotion_model",
    "emotion",
    "emotion_train.csv",
    "emotion_val.csv",
    "emotion_test.csv",
    "emotion"
)

print("\n" + "=" * 60)
print("ALL 3 MODELS TRAINED SUCCESSFULLY")
print("=" * 60)

for f in os.listdir(MODEL_DIR):
    if f.endswith(".joblib"):
        print(os.path.join(MODEL_DIR, f))
