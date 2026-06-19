import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from feature_engineering import DEFAULT_BANDS, FS, prepare_feature_matrix

def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/eeg_data.csv"
    print("Trying to load EEG dataset from GitHub mirror...")
    try:
        df = pd.read_csv(url)
        print(f"[OK] Dataset loaded from GitHub. Shape: {df.shape}")
        if 'class' in df.columns:
            y_binary = (df['class'] == 1).astype(int).values
            X = df.drop('class', axis=1)
        elif 'y' in df.columns:
            y_binary = (df['y'] == 1).astype(int).values
            X = df.drop('y', axis=1)
        else:
            X = df.iloc[:, :-1]
            y_raw = df.iloc[:, -1]
            y_binary = (y_raw == 1).astype(int).values
    except Exception as e:
        print(f"[WARN] Failed to load dataset: {e}")
        print("Creating synthetic dataset for demonstration purposes...")
        np.random.seed(42)
        n_samples = 500
        n_points = 178
        X_synthetic = []
        y_synthetic = []
        for i in range(n_samples):
            if i < 100:
                t = np.linspace(0, 1, n_points)
                signal = 5 * np.sin(2 * np.pi * 8 * t) + 3 * np.sin(2 * np.pi * 4 * t)
                signal += np.random.normal(0, 0.5, n_points)
                y_synthetic.append(1)
            else:
                t = np.linspace(0, 1, n_points)
                signal = 0.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 1, n_points)
                y_synthetic.append(0)
            X_synthetic.append(signal)
        X = pd.DataFrame(X_synthetic)
        y_binary = np.array(y_synthetic)
        print("[OK] Synthetic dataset created successfully.")
    return X, y_binary

def main():
    print("=" * 60)
    print("EEG SEIZURE DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    X_raw, y = load_data()
    print(f"\nExtracting features from {len(X_raw)} EEG samples...")
    feature_df = prepare_feature_matrix(X_raw, y, DEFAULT_BANDS, FS)
    print(f"Feature matrix shape: {feature_df.shape}")
    X_feat = feature_df.drop('target', axis=1)
    y_feat = feature_df['target']
    X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_feat, y_feat, test_size=0.2, stratify=y_feat, random_state=42
    )
    test_indices = X_test_feat.index
    X_test_signals = X_raw.loc[test_indices].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    print("\nTraining Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    print("[OK] Model training complete!")
    y_pred = gb_model.predict(X_test_scaled)
    y_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
    accuracy = gb_model.score(X_test_scaled, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Non-Seizure', 'Seizure']))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Precision-Recall AUC (AP): {pr_auc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("="*50)
    model_filepath = 'eeg_model.pkl'
    print(f"\nSaving model bundle to {model_filepath}...")
    model_bundle = {
        'model': gb_model,
        'scaler': scaler,
        'feature_names': X_feat.columns.tolist(),
        'bands': DEFAULT_BANDS,
        'fs': FS,
        'metrics': {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        },
        'test_data': {
            'raw_signals': X_test_signals.tolist(),
            'features': X_test_feat.values.tolist(),
            'features_df': X_test_feat,
            'scaled_features': X_test_scaled.tolist(),
            'labels': y_test.values.tolist()
        }
    }
    joblib.dump(model_bundle, model_filepath)
    print(f"[OK] Saved model bundle successfully! File size: {os.path.getsize(model_filepath) / 1024:.2f} KB")

if __name__ == '__main__':
    main()