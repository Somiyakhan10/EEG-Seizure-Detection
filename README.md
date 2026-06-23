# EEG Seizure Detection using Ensemble Learning and Explainable AI

This project presents an automated EEG seizure detection system that combines signal processing, ensemble machine learning, and Explainable AI (XAI) techniques. The model analyzes EEG signals and classifies them as Seizure or Non-Seizure while providing insights into the factors influencing predictions.

## Features

* EEG Signal Processing and Analysis
* Feature Extraction from EEG Signals
* Ensemble Machine Learning Models
* Explainable AI (XAI)
* Seizure vs Non-Seizure Classification
* Model Evaluation and Visualization

## Results

### EEG Signal Visualization

This plot shows the raw EEG waveform, highlighting variations in brain electrical activity over time.

<img width="1733" height="773" alt="image" src="https://github.com/user-attachments/assets/90914ef4-d6ee-4c81-be40-a03f121927c8" />

### Power Spectral Density

This visualization illustrates the distribution of signal power across different frequency bands of the EEG signal.

<img width="1726" height="683" alt="image" src="https://github.com/user-attachments/assets/9f0ff97f-d983-42e6-83e2-05df631b8f1f" />

### Confusion Matrix

This matrix summarizes classification performance by comparing actual and predicted seizure labels.

<img width="867" height="589" alt="image" src="https://github.com/user-attachments/assets/64d4551e-1765-4cf4-adca-857dfadf3356" />

### ROC Curve

The ROC curve evaluates the model's ability to distinguish between seizure and non-seizure classes.

<img width="881" height="585" alt="image" src="https://github.com/user-attachments/assets/da6dbc6c-637d-4914-8eaa-422ebe3262e1" />

### Feature Importance

This chart highlights the most influential features used by the ensemble model for seizure prediction.

<img width="1730" height="628" alt="image" src="https://github.com/user-attachments/assets/6cc2eb54-0b1f-4854-95c7-8fc8eb31857c" />

### SHAP Explanation

SHAP visualizations provide model interpretability by showing how individual features contribute to predictions.

<img width="1250" height="791" alt="image" src="https://github.com/user-attachments/assets/e32a7cb6-1402-4b5f-b518-77976ffce682" />

<img width="1239" height="779" alt="image" src="https://github.com/user-attachments/assets/ba6df976-1209-4502-b3f4-63a34110024a" />

### Seizure Alert Simulation

This simulation demonstrates how the system generates real-time alerts when seizure activity is detected.

<img width="1737" height="437" alt="image" src="https://github.com/user-attachments/assets/7eeb5432-b3f3-4c46-80c5-257af6dd1d60" />

### Patient-Level Seizure Risk

This dashboard estimates seizure risk at the patient level based on aggregated EEG analysis.

<img width="1738" height="477" alt="image" src="https://github.com/user-attachments/assets/ce1f0185-a566-4778-8e04-8c9e11f1cf82" />

### Seizure Detection Timeline

This timeline visualizes the occurrence and progression of detected seizure events over time.

<img width="1724" height="400" alt="image" src="https://github.com/user-attachments/assets/3d8431eb-75a0-452b-ab42-49d42425730a" />

### Wavelet Scalograms

Wavelet scalograms reveal the time-frequency characteristics of EEG signals, helping identify seizure patterns.

<img width="1733" height="687" alt="image" src="https://github.com/user-attachments/assets/1925ad50-09bb-481e-96bc-a24d4ad7c8e3" />

### Feature Domains

This visualization compares temporal, spectral, and wavelet-domain features extracted from EEG signals.

<img width="1450" height="758" alt="image" src="https://github.com/user-attachments/assets/b31fc6d3-22b2-41aa-99fe-a8862aeb4c50" />

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* SHAP
* Jupyter Notebook

## Installation

```bash
pip install -r requirements.txt
jupyter notebook EEG_Seizure_Detection_Ensemble_XAI.ipynb
```
