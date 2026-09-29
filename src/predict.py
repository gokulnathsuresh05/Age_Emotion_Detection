import os
import sys
import numpy as np
import librosa
import joblib

PROJECT = r"E:\Project\Age_Emotion_Detection"
MODEL_DIR = os.path.join(PROJECT, "models")

GENDER_MODEL = os.path.join(MODEL_DIR, "gender_model.joblib")
AGE_MODEL = os.path.join(MODEL_DIR, "age_model.joblib")
EMOTION_MODEL = os.path.join(MODEL_DIR, "emotion_model.joblib")


def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True, duration=4.0)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    target = 16000 * 4

    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        [np.mean(zcr), np.std(zcr)],
        [np.mean(centroid), np.std(centroid)],
        [np.mean(bandwidth), np.std(bandwidth)],
        [np.mean(rolloff), np.std(rolloff)]
    ])

    return features.astype(np.float32).reshape(1, -1)


def main():

    if len(sys.argv) < 2:
        print("\nUsage:")
        print(r'python src\predict.py "test_audio\voice.wav"')
        return

    audio_path = sys.argv[1]

    if not os.path.isfile(audio_path):
        print("\nERROR: Audio file not found:")
        print(audio_path)
        return

    print("\n" + "=" * 60)
    print("AGE AND EMOTION DETECTION")
    print("=" * 60)

    print("\nLoading models...")

    gender_model = joblib.load(GENDER_MODEL)
    age_model = joblib.load(AGE_MODEL)
    emotion_model = joblib.load(EMOTION_MODEL)

    print("Models loaded successfully.")

    print("\nProcessing audio:")
    print(audio_path)

    X = extract_features(audio_path)

    print("Feature shape:", X.shape)

    # --------------------------------------------------
    # STEP 1: GENDER
    # --------------------------------------------------

    gender = str(gender_model.predict(X)[0])

    print("\nGender:", gender)

    # Female rejection
    if gender.lower() == "female":

        print("\n" + "-" * 60)
        print("RESULT")
        print("-" * 60)
        print("Upload male voice.")
        print("-" * 60)

        return

    # --------------------------------------------------
    # STEP 2: AGE
    # --------------------------------------------------

    age = str(age_model.predict(X)[0])

    print("Age Group:", age)

    # --------------------------------------------------
    # STEP 3: SENIOR CITIZEN + EMOTION
    # --------------------------------------------------

    if age.lower() == "sixties":

        emotion = str(emotion_model.predict(X)[0])

        print("\n" + "-" * 60)
        print("RESULT")
        print("-" * 60)
        print("Gender        :", gender)
        print("Age Group     :", age)
        print("Status        : Senior Citizen")
        print("Emotion       :", emotion)
        print("-" * 60)

    else:

        print("\n" + "-" * 60)
        print("RESULT")
        print("-" * 60)
        print("Gender        :", gender)
        print("Age Group     :", age)
        print("Status        : Below Senior Age Category")
        print("Emotion       : Not Required")
        print("-" * 60)


if __name__ == "__main__":
    main()
