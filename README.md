#  NeuroDetect: EEG Seizure Detection Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


##  Overview

**NeuroDetect** is a computational neuroscience and biomedical engineering platform that applies machine learning to automatically detect epileptic seizures from EEG (electroencephalogram) signals. The system combines advanced signal processing techniques (wavelet transforms, bandpass filtering, power spectral density analysis) with a Gradient Boosting classifier to distinguish between ictal (seizure) and inter-ictal (normal) brain states.

### Why This Matters
- **50 million people** worldwide live with epilepsy
- **Manual EEG analysis** is time-consuming and subjective
- **Early detection** significantly improves patient outcomes
- **AI-powered tools** can reduce neurologist workload

---

##  Dashboard Overview

<div align="center">
  <img width="959" height="437" alt="Dashboard Overview" src="https://github.com/user-attachments/assets/a9ed4acb-5373-4a37-a16c-2bc2ad6b5bb5" />
  <br/>
  <em>Figure 1: The NeuroDetect main dashboard showing the biomedical EEG seizure detection platform interface.</em>
</div>

The main dashboard provides a comprehensive overview of the platform with:
- **Navigation Sidebar**: Quick access to all 5 pages (Home, EEG Analysis, Seizure Prediction, Explainable AI, Model Performance)
- **Performance Metrics**: Real-time display of model accuracy, sensitivity, specificity, and ROC-AUC scores
- **System Status**: Indicates whether the model is loaded and ready for analysis
- **Professional Dark Theme**: Clinical-grade dark blue interface optimized for extended use

---

##  EEG Signal Analysis

<div align="center">
  <img width="1912" height="859" alt="EEG Signal Analysis" src="https://github.com/user-attachments/assets/ce296586-8657-4764-8d5f-a93e47eac2e4" />
  <br/>
  <em>Figure 2: Waveform decomposition showing the raw EEG signal and its frequency bands (Delta, Theta, Alpha, Beta, Gamma).</em>
</div>

The EEG Analysis page provides interactive signal visualization with three tabs:

### 1. Waveform Decomposition
- Displays the raw EEG signal at the top
- Shows filtered signals for each frequency band below
- Delta (0.5-4 Hz): Deep sleep, pathological activity
- Theta (4-8 Hz): Drowsiness, hippocampal rhythms
- Alpha (8-13 Hz): Relaxed wakefulness
- Beta (13-30 Hz): Active cognition, cortical hyperexcitability
- Gamma (30-45 Hz): High-frequency discharges, ictal activity

### 2. Power Spectral Density (PSD)
- Shows energy distribution across frequencies
- Identifies dominant frequency bands
- Highlights seizure-related spectral patterns

### 3. Wavelet Scalogram
- Continuous wavelet transform visualization
- Time-frequency representation of EEG signals
- Captures transient seizure spikes

---

##  Seizure Prediction

<div align="center">
  <img width="1907" height="883" alt="Seizure Prediction" src="https://github.com/user-attachments/assets/c2026623-bf97-4f94-a453-613ddb9defe8" />
  <br/>
  <em>Figure 3: Real-time seizure prediction with probability gauge and frequency band analysis.</em>
</div>

The Seizure Prediction page enables real-time diagnostics:

### Input Options
- **Synthetic Signal**: Generate custom EEG signals with adjustable parameters
  - Normal vs Seizure signal types
  - Adjustable noise levels (0.1-3.0)
  - Amplitude control
- **Upload CSV**: Upload custom EEG recordings (178 data points)

### Prediction Output
- **Probability Gauge**: Visual indicator of seizure risk (0-100%)
- **Risk Assessment**: Categorized risk levels (High, Moderate, Low)
- **Clinical Report**: Structured diagnostic summary with recommendations
- **Frequency Band Analysis**: Bar chart showing relative power distribution

---

##  Explainable AI

<div align="center">
  <img width="1846" height="800" alt="Explainable AI" src="https://github.com/user-attachments/assets/d22880b3-cc8e-4b93-aba6-a5b594c02809" />
  <br/>
  <em>Figure 4: Feature importance and SHAP analysis showing which features drive seizure detection.</em>
</div>

The Explainable AI page provides model interpretability:

### Global Explanations
- **Gini Feature Importance**: Top 15 features ranked by importance
- **SHAP Summary Plot**: Global feature impact analysis
  - Red dots: High feature values
  - Blue dots: Low feature values
  - Right side: Higher seizure risk

### Local Explanations
- **SHAP Waterfall Plot**: Individual prediction breakdown
- Shows how each feature contributed to the final prediction
- Red bars: Features pushing toward seizure
- Blue bars: Features pushing toward normal
- Final confidence score displayed

---

##  Model Performance

<div align="center">
  <img width="1904" height="872" alt="Model Performance 1" src="https://github.com/user-attachments/assets/74824c77-3e4b-41e5-b408-89efb45523f9" />
  <br/>
  <em>Figure 5: Model performance metrics including confusion matrix and evaluation curves.</em>
</div>

<div align="center">
  <img width="1876" height="804" alt="Model Performance 2" src="https://github.com/user-attachments/assets/197b2c18-9f0c-4ff8-bd94-8ed50ec33f1b" />
  <br/>
  <em>Figure 6: ROC curve showing the model's diagnostic performance.</em>
</div>

<div align="center">
  <img width="1888" height="807" alt="Model Performance 3" src="https://github.com/user-attachments/assets/acf9f52b-52c8-49f3-9d65-818ac5434cdb" />
  <br/>
  <em>Figure 7: Confusion matrix showing detailed classification results.</em>
</div>

The Model Performance page provides comprehensive evaluation:

### Key Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 94.0% |
| **Sensitivity (Recall)** | 80.0% |
| **Specificity** | 97.5% |
| **F1-Score** | 84.2% |

### Visualizations
- **Confusion Matrix**: Detailed breakdown of predictions
  - True Negatives (TN): Correctly identified normal signals
  - False Positives (FP): Normal incorrectly flagged as seizure
  - False Negatives (FN): Seizures missed by the system
  - True Positives (TP): Seizures correctly identified

- **ROC Curve**: Sensitivity vs 1-Specificity
  - AUC Score: 0.957 (Excellent discrimination)
  - Shows model's diagnostic performance

- **Precision-Recall Curve**: Precision vs Recall
  - PR-AUC: 0.900
  - Critical for imbalanced datasets

---

##  Features

### Signal Processing
- Bandpass filtering isolates 5 physiological frequency bands
- Wavelet transform captures transient seizure spikes
- Power Spectral Density analysis for frequency domain insights

### Feature Engineering
- **Statistical**: Mean, std, variance, skewness, kurtosis, RMS
- **Spectral**: Entropy, band powers, relative power ratios
- **Wavelet**: Energy at decomposition levels, wavelet entropy

### Machine Learning
- **Algorithm**: Gradient Boosting Classifier
- **Validation**: 5-fold Stratified Cross-Validation
- **Preprocessing**: Standard scaling for feature normalization
- **Interpretability**: SHAP for model explanations

---

##  Technology Stack

| Component | Technology |
|-----------|------------|
| **Core Language** | Python 3.8+ |
| **Signal Processing** | SciPy, PyWavelets, NumPy |
| **Machine Learning** | Scikit-learn (Gradient Boosting) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Explainable AI** | SHAP |
| **Model Persistence** | Joblib |
| **Data Manipulation** | Pandas |

---

##  Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurodetect.git
cd neurodetect

# Install dependencies
pip install -r requirements.txt

# Train the model
python model_training.py

# Launch the dashboard
streamlit run app.py
