# Classify Genre of Music Using Random Forests

**COMP_SCI 352 — Machine Perception of Music & Audio**  
Northwestern University  

**Authors:**  
Katerina Falkner  
Ishani Pidara  

Link: https://ipidara.github.io/genre_identifier/

---

## Overview

This project builds a machine learning model that classifies the genre of a music audio clip using a **Random Forest classifier** trained on the **GTZAN dataset**. We also analyze dataset limitations, feature importance, and the effect of duplicate audio samples on model performance.

---

## Dataset

We used the **GTZAN dataset**, which contains:

- 1000 audio clips
- 10 music genres
- 30-second samples
- 59 extracted audio features (MFCCs, chroma, spectral features, etc.)

Because GTZAN contains known duplicate recordings, we evaluated two versions:

1. Original dataset  
2. Modified dataset with 51 duplicate files removed

---

## Methodology

- Implemented using `sklearn.RandomForestClassifier`
- Data split: **80% training, 20% testing**
- Predictions generated using majority voting across decision trees

### Evaluation Metrics
- Accuracy score
- Confusion matrix
- Feature importance

---

## Results

| Dataset | Accuracy |
|--------|---------|
| Original GTZAN | 75% |
| Modified GTZAN | 78.42% |

### Conclusions
- Removing duplicates improved overall accuracy.
- Classical music was easiest to classify.
- Rock and pop were frequently misclassified.
- Harmonic and spectral features were most important for prediction.

