import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import joblib
import pygame

BASE_DIR = r"E:\Project\Age_Emotion_Detection"
MODEL_DIR = os.path.join(BASE_DIR, "models")

GENDER_MODEL_PATH = os.path.join(MODEL_DIR, "gender_model.joblib")
AGE_MODEL_PATH = os.path.join(MODEL_DIR, "age_model.joblib")
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.joblib")

# Load models
gender_model = joblib.load(GENDER_MODEL_PATH)
age_model = joblib.load(AGE_MODEL_PATH)
emotion_model = joblib.load(EMOTION_MODEL_PATH)

pygame.mixer.init()

current_audio = None
is_paused = False


def extract_features(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
        duration=4
    )

    target_length = sr * 4

    if len(y) < target_length:
        y = np.pad(
            y,
            (0, target_length - len(y))
        )
    else:
        y = y[:target_length]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfcc)

    zcr = librosa.feature.zero_crossing_rate(y)

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(delta, axis=1),
        np.std(delta, axis=1),

        np.mean(zcr),
        np.std(zcr),

        np.mean(centroid),
        np.std(centroid),

        np.mean(bandwidth),
        np.std(bandwidth),

        np.mean(rolloff),
        np.std(rolloff)
    ])

    return features.reshape(1, -1)


def choose_audio():
    global current_audio

    path = filedialog.askopenfilename(
        title="Select Voice File",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
    )

    if not path:
        return

    current_audio = path

    file_label.config(
        text=os.path.basename(path)
    )

    path_label.config(
        text=path
    )

    result_text.delete(
        "1.0",
        tk.END
    )

    result_text.insert(
        tk.END,
        "Voice selected successfully.\n\n"
        "Click PREDICT."
    )

    status_label.config(
        text="Voice selected."
    )


def play_voice():
    global is_paused

    if not current_audio:
        messagebox.showwarning(
            "No Voice",
            "Please select a voice file first."
        )
        return

    try:
        pygame.mixer.music.load(current_audio)
        pygame.mixer.music.play()

        is_paused = False

        status_label.config(
            text="Playing voice..."
        )

    except Exception as e:
        messagebox.showerror(
            "Playback Error",
            str(e)
        )


def pause_resume_voice():
    global is_paused

    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False

        status_label.config(
            text="Playing voice..."
        )

    elif pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
        is_paused = True

        status_label.config(
            text="Voice paused."
        )


def stop_voice():
    global is_paused

    pygame.mixer.music.stop()
    is_paused = False

    status_label.config(
        text="Playback stopped."
    )


def predict_voice():

    if not current_audio:
        messagebox.showwarning(
            "No Voice",
            "Please select a voice file first."
        )
        return

    if not os.path.exists(current_audio):
        messagebox.showerror(
            "File Error",
            "Selected audio file does not exist."
        )
        return

    try:

        status_label.config(
            text="Processing voice... Please wait."
        )

        root.update()

        # Extract 168 features
        features = extract_features(
            current_audio
        )

        if features.shape[1] != 168:
            raise ValueError(
                f"Expected 168 features, got {features.shape[1]}"
            )

        # Gender prediction
        gender = str(
            gender_model.predict(features)[0]
        ).lower()

        # Female rejection
        if gender == "female":

            result_text.delete(
                "1.0",
                tk.END
            )

            result_text.insert(
                tk.END,
                "========================================\n"
                "             RESULT\n"
                "========================================\n\n"
                "Gender : Female\n\n"
                "Upload male voice.\n\n"
                "========================================"
            )

            status_label.config(
                text="Female voice detected."
            )

            return

        # Male age prediction
        age_group = str(
            age_model.predict(features)[0]
        ).lower()

        # Senior Citizen
        if age_group == "sixties":

            emotion = str(
                emotion_model.predict(features)[0]
            ).lower()

            result_text.delete(
                "1.0",
                tk.END
            )

            result_text.insert(
                tk.END,
                "========================================\n"
                "             RESULT\n"
                "========================================\n\n"
                f"Gender           : Male\n"
                f"Age Group        : {age_group.title()}\n"
                f"Status           : Senior Citizen\n"
                f"Emotion Required : YES\n"
                f"Detected Emotion : {emotion.title()}\n\n"
                "========================================"
            )

            status_label.config(
                text=f"Emotion detected: {emotion.title()}"
            )

        # Non-senior
        else:

            result_text.delete(
                "1.0",
                tk.END
            )

            result_text.insert(
                tk.END,
                "========================================\n"
                "             RESULT\n"
                "========================================\n\n"
                f"Gender           : Male\n"
                f"Age Group        : {age_group.title()}\n"
                f"Status           : Below Senior Age Category\n"
                f"Emotion Required : NO\n"
                f"Emotion          : Not Required\n\n"
                "========================================"
            )

            status_label.config(
                text="Age detected. Emotion not required."
            )

    except Exception as e:

        result_text.delete(
            "1.0",
            tk.END
        )

        result_text.insert(
            tk.END,
            "PROCESSING ERROR\n\n"
            f"{type(e).__name__}: {e}"
        )

        status_label.config(
            text="Processing failed."
        )

        messagebox.showerror(
            "Processing Error",
            f"{type(e).__name__}:\n\n{e}"
        )


def clear_all():
    global current_audio
    global is_paused

    pygame.mixer.music.stop()

    current_audio = None
    is_paused = False

    file_label.config(
        text="No voice selected"
    )

    path_label.config(
        text="No file selected"
    )

    result_text.delete(
        "1.0",
        tk.END
    )

    result_text.insert(
        tk.END,
        "Select a voice file to begin."
    )

    status_label.config(
        text="Ready."
    )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Age and Emotion Detection Through Voice"
)

root.geometry(
    "820x760"
)

root.resizable(
    False,
    False
)

# Title
tk.Label(
    root,
    text="AGE AND EMOTION DETECTION",
    font=("Arial", 22, "bold")
).pack(
    pady=(20, 5)
)

tk.Label(
    root,
    text="Voice-based Male Gender, Age and Emotion Detection",
    font=("Arial", 11)
).pack(
    pady=(0, 20)
)

# Select
tk.Button(
    root,
    text="SELECT VOICE",
    command=choose_audio,
    width=20,
    height=2,
    font=("Arial", 11, "bold")
).pack(
    pady=5
)

file_label = tk.Label(
    root,
    text="No voice selected",
    font=("Arial", 11)
)

file_label.pack(
    pady=5
)

path_label = tk.Label(
    root,
    text="No file selected",
    font=("Arial", 8),
    wraplength=760
)

path_label.pack(
    pady=(0, 15)
)

# Playback buttons
play_frame = tk.Frame(root)

play_frame.pack(
    pady=5
)

tk.Button(
    play_frame,
    text="PLAY",
    command=play_voice,
    width=12,
    height=2
).grid(
    row=0,
    column=0,
    padx=5
)

tk.Button(
    play_frame,
    text="PAUSE / RESUME",
    command=pause_resume_voice,
    width=18,
    height=2
).grid(
    row=0,
    column=1,
    padx=5
)

tk.Button(
    play_frame,
    text="STOP",
    command=stop_voice,
    width=12,
    height=2
).grid(
    row=0,
    column=2,
    padx=5
)

# Predict
tk.Button(
    root,
    text="PREDICT",
    command=predict_voice,
    width=25,
    height=2,
    font=("Arial", 12, "bold")
).pack(
    pady=(20, 8)
)

# Clear
tk.Button(
    root,
    text="CLEAR",
    command=clear_all,
    width=25,
    height=2
).pack(
    pady=5
)

# Result title
tk.Label(
    root,
    text="DETECTION RESULT",
    font=("Arial", 14, "bold")
).pack(
    pady=(20, 8)
)

# Result box
result_text = tk.Text(
    root,
    height=12,
    width=82,
    font=("Consolas", 11),
    wrap=tk.WORD
)

result_text.pack(
    padx=20
)

result_text.insert(
    tk.END,
    "Select a voice file to begin."
)

# Status
status_label = tk.Label(
    root,
    text="Ready.",
    font=("Arial", 10)
)

status_label.pack(
    pady=10
)

# Dataset information
tk.Label(
    root,
    text="Dataset: F:\\Age_Emotion_Detection_Dataset\\clips",
    font=("Arial", 8),
    wraplength=760
).pack(
    pady=5
)

root.mainloop()

pygame.mixer.quit()
