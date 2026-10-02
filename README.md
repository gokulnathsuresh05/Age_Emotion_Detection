# Age and Emotion Detection Through Voice

A Machine Learning project that detects gender, age group, senior-citizen status, and emotion from a voice recording.

## Project Workflow

Voice Input
↓
Audio Feature Extraction
↓
Gender Detection
↓
Female → Upload male voice.
↓
Male
↓
Age Group Detection
↓
Sixties → Senior Citizen → Emotion Detection
↓
Other Age Group → Age Prediction Only

## Features

- Male/Female voice classification
- Age-group prediction
- Senior Citizen identification
- Emotion detection for senior-age category
- Female voice rejection
- Audio playback
- GUI-based prediction

## Machine Learning Models

### Gender Detection
Random Forest Classifier

### Age Detection
Random Forest Classifier

### Emotion Detection
Support Vector Machine (SVM) with RBF kernel and StandardScaler

## Audio Features

The system extracts 168 audio features using:

- MFCC mean
- MFCC standard deviation
- Delta MFCC mean
- Delta MFCC standard deviation
- Zero Crossing Rate
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff

Audio is converted to mono and processed at 16 kHz.

## Datasets

The project uses:

- Mozilla Common Voice – South Asian English
- RAVDESS
- CREMA-D

The original datasets, audio files, and generated feature files are not included in this repository because of their large size.

## Model Results

| Model | Test Accuracy |
|---|---:|
| Gender | 97.32% |
| Age Group | 67.66% |
| Emotion | 55.87% |

These results are based on the project's test datasets and should not be considered guaranteed real-world accuracy.

## Important Age Note

## Dataset Sources

The project uses the following publicly available datasets:

1. Mozilla Common Voice – South Asian English
   - Used for age-group and gender classification.
   - Dataset: https://datacollective.mozillafoundation.org/

2. RAVDESS
   - Used for speech emotion classification.
   - Dataset: https://zenodo.org/records/1188976

3. CREMA-D
   - Used for speech emotion classification.
   - Dataset: https://github.com/CheyneyComputerScience/CREMA-D

Sample dataset drive link 
google_drive = https://drive.google.com/file/d/1JijCGAcC34LnGrj5mAA5mrkjexIoakb7/view?usp=sharing
     
## Project Structure

Age_Emotion_Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── gender_model.joblib
│   ├── age_model.joblib
│   └── emotion_model.joblib
│
└── src/
    ├── predict.py
    ├── preprocess.py
    └── train_models.py

## Installation

Create a virtual environment:

python -m venv .venv

Activate it in Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## Run the Application

python app.py

## Run Prediction from Terminal

python src/predict.py

## Note

The dataset and audio files are excluded from GitHub using .gitignore.

## Author

Gokulnath S
