import os
import numpy as np
import pandas as pd
import librosa

BASE = r"F:\Age_Emotion_Detection_Dataset"
CLIPS = os.path.join(BASE, "clips")
FEATURES = os.path.join(BASE, "features")


def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True, duration=4.0)

    if len(y) == 0:
        raise ValueError("Empty audio")

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

    return features.astype(np.float32)


def process(csv_path, output_path, audio_column):
    df = pd.read_csv(csv_path)

    X = []
    valid_rows = []
    failed = 0

    print("\nProcessing:", csv_path)
    print("Total files:", len(df))

    for i, row in df.iterrows():

        if audio_column == "filename":
            audio_path = os.path.join(CLIPS, str(row["filename"]))
        else:
            audio_path = str(row["path"])

        try:
            if not os.path.isfile(audio_path):
                failed += 1
                continue

            feature = extract_features(audio_path)
            X.append(feature)
            valid_rows.append(row.to_dict())

        except Exception:
            failed += 1

        if (i + 1) % 500 == 0:
            print("Processed:", i + 1, "/", len(df))

    X = np.asarray(X, dtype=np.float32)
    valid_df = pd.DataFrame(valid_rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, X)

    label_path = os.path.splitext(output_path)[0] + "_labels.csv"
    valid_df.to_csv(label_path, index=False)

    print("SUCCESS:", os.path.basename(output_path))
    print("Feature shape:", X.shape)
    print("Valid:", len(X))
    print("Failed:", failed)


def main():

    print("=" * 60)
    print("AGE AND EMOTION DETECTION - AUDIO PREPROCESSING")
    print("=" * 60)

    datasets = [
        ("gender", "gender_train.csv", "filename"),
        ("gender", "gender_val.csv", "filename"),
        ("gender", "gender_test.csv", "filename"),

        ("age", "age_train.csv", "filename"),
        ("age", "age_val.csv", "filename"),
        ("age", "age_test.csv", "filename"),

        ("emotion", "emotion_train.csv", "path"),
        ("emotion", "emotion_val.csv", "path"),
        ("emotion", "emotion_test.csv", "path")
    ]

    for category, filename, column in datasets:

        if category == "emotion":
            csv_path = os.path.join(
                BASE,
                "emotion",
                "final_prepare",
                filename
            )
        else:
            csv_path = os.path.join(
                BASE,
                "final_prepare",
                filename
            )

        output_dir = os.path.join(FEATURES, category)

        output_path = os.path.join(
            output_dir,
            filename.replace(".csv", "_features.npy")
        )

        process(csv_path, output_path, column)

    print("\n" + "=" * 60)
    print("ALL AUDIO PREPROCESSING COMPLETED")
    print("FEATURES:", FEATURES)
    print("=" * 60)


if __name__ == "__main__":
    main()
