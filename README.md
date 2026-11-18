# 🎙️ EMOTION AI — Voice Emotion Intelligence System

**An AI-powered speech emotion recognition system built using HuBERT and Machine Learning.**

EMOTION AI analyzes speech recordings and predicts the vocal emotion expressed in the audio. It uses pretrained speech representations from HuBERT and a Logistic Regression classifier to identify one of eight emotion categories.

## 🚀 Project Overview

Human speech contains more than words. Tone, rhythm, pitch, and other acoustic patterns can provide useful signals about vocal expression.

EMOTION AI explores how machine learning can learn these patterns and classify speech into emotion categories.

**Supported emotion categories:**
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

> Note: The system predicts vocal emotion labels. It does not determine a person's actual internal emotional or psychological state.

## 🧠 How It Works

1. **Audio Input:** The user provides a speech recording.
2. **Preprocessing:** Audio is loaded and resampled to 16 kHz.
3. **Feature Extraction:** A pretrained HuBERT model extracts speech representations.
4. **Pooling:** Frame-level representations are mean-pooled into a 768-dimensional feature vector.
5. **Classification:** A trained Logistic Regression classifier predicts the emotion category.
6. **Visualization:** The Gradio interface displays the predicted category and class scores.

## 🏗️ Architecture

```text
             Audio Input
                  |
                  v
          Audio Preprocessing
             (16 kHz)
                  |
                  v
          Pretrained HuBERT
          (Speech Encoder)
                  |
                  v
          Mean Pooling
           (768 features)
                  |
                  v
       Logistic Regression
           Classifier
                  |
                  v
         Emotion Prediction
                  |
                  v
        Gradio Web Interface
```

## 📊 Model Performance

The final noise-augmented HuBERT classifier was evaluated on 300 RAVDESS test recordings from five held-out actors.

| Metric | Result |
|---|---:|
| Test Accuracy | **57.33%** |
| Macro F1-score | **0.5775** |
| Weighted F1-score | **0.5869** |
| Correct predictions | 172 / 300 |

### Cross-validation experiments

Five-fold actor-grouped cross-validation on the training data produced the following results:

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Original HuBERT features | 67.67% | 66.54% |
| HuBERT + Noise Augmentation | **69.44%** | **68.47%** |
| HuBERT + Time Stretch | 66.44% | 65.32% |
| HuBERT + Both Augmentations | 68.67% | 67.43% |

**Important:** Cross-validation and final test results measure different evaluation stages. Noise augmentation improved cross-validation performance, but the final test accuracy remained 57.33%. The test set has also been inspected in earlier experiments, so it should not be described as a completely untouched benchmark.

## 🧪 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** speech dataset.

- 1,440 speech recordings
- 24 actors
- 8 emotion categories
- Actor-independent training, validation, and test splits

The dataset is not included in this repository. Users must obtain it separately and follow its **CC BY-NC-SA 4.0** license.

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- HuBERT (`facebook/hubert-base-ls960`)
- Scikit-learn
- Librosa
- NumPy
- Joblib
- Gradio

## 📂 Project Structure

```text
EMOTION_AI/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── hubert_noise_augmented_classifier.joblib
│   └── hubert_noise_augmented_model_info.json
│
└── src/
    ├── inference.py
    ├── train_final_emotion_classifier.py
    ├── extract_hubert_features.py
    ├── extract_hubert_augmented_features.py
    ├── cross_validate_hubert_augmentation.py
    └── compare_hubert_augmentation_types.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Albin2003-maker/EMOTION_AI.git
cd EMOTION_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your hardware if the pinned version in `requirements.txt` is unavailable or unsuitable.

### 4. Run the application

```bash
python app.py
```

Open the local Gradio URL displayed in your terminal.

**First launch:** HuBERT model weights are downloaded from Hugging Face, so an internet connection is required.

## 🎯 Future Improvements

- Improve recognition of difficult emotion pairs, especially sad and neutral.
- Explore additional speaker-independent datasets.
- Investigate more effective audio augmentation strategies.
- Experiment with fine-tuning pretrained speech models.
- Improve robustness across different speakers and recording conditions.
- Deploy the application on AWS.

## 👨‍💻 Author

**Albin Thomas**

GitHub: [@Albin2003-maker](https://github.com/Albin2003-maker)

---

*Built as an exploration of speech intelligence, representation learning, and machine learning.*