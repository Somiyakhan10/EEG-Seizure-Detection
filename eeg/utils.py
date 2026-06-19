import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

HEX_DARK_BG = "#0A1628"
HEX_SEIZURE_RED = "#FF6B6B"
HEX_NORMAL_GREEN = "#00E676"
HEX_ALERT_ORANGE = "#FFB74D"
HEX_ACCENT_BLUE = "#2A5CAA"
HEX_LIGHT_BLUE = "#90E0EF"

BAND_COLORS = {
    'Delta': '#1F77B4',
    'Theta': '#FF7F0E',
    'Alpha': '#2CA02C',
    'Beta': '#D62728',
    'Gamma': '#9467BD'
}

def plot_raw_and_filtered_signals(raw_signal, fs, bands_dict):
    try:
        from feature_engineering import apply_bandpass_filter
        t = np.arange(len(raw_signal)) / fs
        n_bands = len(bands_dict)
        fig = make_subplots(
            rows=n_bands + 1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=["Original EEG Signal (Raw)"] + [f"{name} Band ({low}-{high} Hz)" for name, (low, high) in bands_dict.items()]
        )
        fig.add_trace(
            go.Scatter(x=t, y=raw_signal, name="Raw Signal", line=dict(color='white', width=1.5)),
            row=1, col=1
        )
        for idx, (band_name, (low, high)) in enumerate(bands_dict.items(), start=2):
            filtered = apply_bandpass_filter(raw_signal, low, high, fs)
            fig.add_trace(
                go.Scatter(x=t, y=filtered, name=band_name, line=dict(color=BAND_COLORS[band_name], width=1.2)),
                row=idx, col=1
            )
        fig.update_layout(
            template="plotly_dark",
            height=180 * (n_bands + 1),
            showlegend=False,
            margin=dict(l=30, r=20, t=50, b=30),
            title_text="EEG Signal Decomposition into Physiological Bands",
            title_x=0.5,
            paper_bgcolor='#0A1628',
            plot_bgcolor='#0A1628'
        )
        fig.update_yaxes(title_text="Amp (μV)", row=1, col=1)
        for idx in range(2, n_bands + 2):
            fig.update_yaxes(title_text="Amp (μV)", row=idx, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=n_bands + 1, col=1)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_single_psd(raw_signal, fs, bands_dict):
    try:
        nperseg = min(128, len(raw_signal))
        freqs, psd = scipy_signal.welch(raw_signal, fs, nperseg=nperseg)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs, y=psd,
            mode='lines',
            line=dict(color=HEX_ACCENT_BLUE, width=2.5),
            name='PSD'
        ))
        annotations = []
        for band_name, (low, high) in bands_dict.items():
            if high < fs/2:
                fig.add_shape(
                    type="rect",
                    x0=low, x1=high,
                    y0=0, y1=1,
                    yref="paper",
                    fillcolor=BAND_COLORS[band_name],
                    opacity=0.12,
                    layer="below",
                    line=dict(width=0)
                )
                annotations.append(dict(
                    x=(low + high)/2,
                    y=0.95,
                    yref="paper",
                    text=band_name,
                    showarrow=False,
                    font=dict(color="gray", size=10),
                    xanchor="center"
                ))
        fig.update_layout(
            template="plotly_dark",
            title="Power Spectral Density (PSD) Analysis",
            title_x=0.5,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density (μV²/Hz)",
            yaxis_type="log",
            xaxis_range=[0, min(50, fs/2)],
            margin=dict(l=50, r=20, t=50, b=50),
            height=450,
            annotations=annotations,
            paper_bgcolor='#0A1628',
            plot_bgcolor='#0A1628'
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_psd_comparison(seizure_signal, non_seizure_signal, fs, bands_dict):
    try:
        nperseg = min(128, len(seizure_signal))
        f_sz, psd_sz = scipy_signal.welch(seizure_signal, fs, nperseg=nperseg)
        f_nsz, psd_nsz = scipy_signal.welch(non_seizure_signal, fs, nperseg=nperseg)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f_sz, y=psd_sz, mode='lines', line=dict(color=HEX_SEIZURE_RED, width=2.5), name='Seizure EEG (Ictal)'))
        fig.add_trace(go.Scatter(x=f_nsz, y=psd_nsz, mode='lines', line=dict(color=HEX_NORMAL_GREEN, width=2.5), name='Non-Seizure EEG (Normal)'))
        annotations = []
        for band_name, (low, high) in bands_dict.items():
            if high < fs/2:
                fig.add_shape(type="rect", x0=low, x1=high, y0=0, y1=1, yref="paper", fillcolor=BAND_COLORS[band_name], opacity=0.08, layer="below", line=dict(width=0))
                annotations.append(dict(x=(low+high)/2, y=0.93, yref="paper", text=band_name, showarrow=False, font=dict(color="gray", size=9), xanchor="center"))
        fig.update_layout(template="plotly_dark", title="PSD Comparison: Seizure vs Non-Seizure EEG", title_x=0.5, xaxis_title="Frequency (Hz)", yaxis_title="Power (μV²/Hz)", yaxis_type="log", xaxis_range=[0, min(50, fs/2)], margin=dict(l=50, r=20, t=50, b=50), height=450, annotations=annotations, paper_bgcolor='#0A1628', plot_bgcolor='#0A1628')
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_wavelet_scalogram(raw_signal, fs):
    try:
        scales = np.arange(1, 40)
        coefficients, freqs = pywt.cwt(raw_signal, scales, 'morl', sampling_period=1/fs)
        magnitude = np.abs(coefficients)
        t = np.arange(len(raw_signal)) / fs
        fig = go.Figure(data=go.Heatmap(z=magnitude, x=t, y=freqs, colorscale='Magma', colorbar=dict(title="Magnitude")))
        fig.update_layout(template="plotly_dark", title="Wavelet Scalogram", title_x=0.5, xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", margin=dict(l=50, r=20, t=50, b=50), height=450, paper_bgcolor='#0A1628', plot_bgcolor='#0A1628')
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_confusion_matrix(cm):
    try:
        z = cm
        x = ['Non-Seizure (Predicted)', 'Seizure (Predicted)']
        y = ['Non-Seizure (Actual)', 'Seizure (Actual)']
        total = np.sum(cm)
        percentages = [[f"{(val/total)*100:.1f}%" for val in row] for row in cm]
        hover_text = []
        for i in range(2):
            row_text = []
            for j in range(2):
                row_text.append(f"Actual: {y[i].split(' ')[0]}<br>Predicted: {x[j].split(' ')[0]}<br>Count: {z[i][j]} ({percentages[i][j]})")
            hover_text.append(row_text)
        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale='Blues',
            text=[[f"<b>{z[i][j]}</b><br>{percentages[i][j]}" for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            hoverinfo="text",
            hovertext=hover_text,
            showscale=False
        ))
        fig.update_layout(template="plotly_dark", title="Confusion Matrix", title_x=0.5, xaxis=dict(side="bottom"), margin=dict(l=50, r=50, t=50, b=50), height=380, width=380, paper_bgcolor='#0A1628', plot_bgcolor='#0A1628')
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_roc_curve(y_true, y_prob):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=HEX_ACCENT_BLUE, width=3), name=f"Gradient Boosting (AUC = {roc_auc:.3f})", fill='tozeroy', fillcolor='rgba(42, 92, 170, 0.1)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', width=1.5, dash='dash'), name='Random Guess'))
        fig.update_layout(template="plotly_dark", title="ROC Curve", title_x=0.5, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Sensitivity)", xaxis_range=[-0.01, 1.01], yaxis_range=[-0.01, 1.01], margin=dict(l=50, r=20, t=50, b=50), height=400, paper_bgcolor='#0A1628', plot_bgcolor='#0A1628')
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_pr_curve(y_true, y_prob):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', line=dict(color=HEX_NORMAL_GREEN, width=3), name=f"PR Curve (AUC = {pr_auc:.3f})", fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)'))
        pos_ratio = sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
        fig.add_trace(go.Scatter(x=[0, 1], y=[pos_ratio, pos_ratio], mode='lines', line=dict(color='gray', width=1.5, dash='dash'), name=f'Baseline ({pos_ratio:.2f})'))
        fig.update_layout(template="plotly_dark", title="Precision-Recall Curve", title_x=0.5, xaxis_title="Recall (Sensitivity)", yaxis_title="Precision (PPV)", xaxis_range=[-0.01, 1.01], yaxis_range=[-0.01, 1.01], margin=dict(l=50, r=20, t=50, b=50), height=400, paper_bgcolor='#0A1628', plot_bgcolor='#0A1628')
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def plot_feature_importance(importances, feature_names, top_n=15):
    try:
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_importances = importances[indices]
        sorted_names = [feature_names[i] for i in indices]
        colors = []
        for name in sorted_names:
            matched = False
            for band in BAND_COLORS.keys():
                if band in name:
                    colors.append(BAND_COLORS[band])
                    matched = True
                    break
            if not matched:
                colors.append(HEX_ACCENT_BLUE)
        fig = go.Figure(go.Bar(
            x=sorted_importances,
            y=sorted_names,
            orientation='h',
            marker=dict(color=colors, line=dict(color='black', width=1)),
            text=[f" {val:.4f}" for val in sorted_importances],
            textposition='outside'
        ))
        fig.update_layout(template="plotly_dark", title=f"Top {top_n} Features Driving Seizure Detection", title_x=0.5, xaxis_title="Gini Impurity Reduction Score", yaxis_title="EEG Feature Name", yaxis=dict(autorange="reversed"), margin=dict(l=150, r=50, t=50, b=50), height=top_n * 35 + 100, paper_bgcolor='#0A1628', plot_bgcolor='#0A1628')
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0A1628')
        return fig

def render_shap_summary_plot(explainer, X_sample, feature_names):
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=HEX_DARK_BG)
    ax.set_facecolor(HEX_DARK_BG)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=15, show=False)
    fig.patch.set_facecolor(HEX_DARK_BG)
    for text in fig.axes[0].texts:
        text.set_color('white')
    fig.axes[0].tick_params(colors='white')
    fig.axes[0].xaxis.label.set_color('white')
    fig.axes[0].yaxis.label.set_color('white')
    fig.axes[0].title.set_color('white')
    if len(fig.axes) > 1:
        fig.axes[1].tick_params(colors='white')
        fig.axes[1].yaxis.label.set_color('white')
    plt.tight_layout()
    return fig

def render_shap_waterfall_plot(explainer, base_value, shap_values, sample_features, feature_names):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=HEX_DARK_BG)
    ax.set_facecolor(HEX_DARK_BG)
    exp = shap.Explanation(values=shap_values, base_values=base_value, data=sample_features, feature_names=feature_names)
    shap.plots.waterfall(exp, max_display=10, show=False)
    fig.patch.set_facecolor(HEX_DARK_BG)
    fig.axes[0].tick_params(colors='white')
    fig.axes[0].xaxis.label.set_color('white')
    fig.axes[0].yaxis.label.set_color('white')
    for text in fig.axes[0].texts:
        text.set_color('white')
    plt.tight_layout()
    return fig

def get_clinical_interpretation(prediction, probability, rel_powers):
    report = {}
    if prediction == 1:
        if probability >= 0.80:
            report['status'] = 'CRITICAL ALERT: SEIZURE DETECTED'
            report['status_color'] = HEX_SEIZURE_RED
            report['risk_level'] = 'HIGH RISK (Ictal Activity)'
            report['summary'] = "Highly synchronized electrophysiological waveforms detected. The signal exhibit high-amplitude rhythmic spike-wave discharges strongly characteristic of an active epileptic seizure (ictal state)."
            report['recommendation'] = "1. Trigger urgent nursing and clinical staff notification.\n2. Ensure patient airway clearance and safe physical environment to prevent trauma.\n3. Prepare immediate pharmacological intervention as per clinical seizure protocols.\n4. Initiate long-term continuous video-EEG monitoring."
        else:
            report['status'] = 'WARNING: ABNORMAL PATHOLOGY DETECTED'
            report['status_color'] = HEX_ALERT_ORANGE
            report['risk_level'] = 'MODERATE RISK (Potential Seizure/Pre-Ictal)'
            report['summary'] = "Substantial high-frequency perturbation and rhythmic alpha-theta structures identified. This waveform behavior corresponds to potential subclinical seizure activity, focal ictal onset, or an evolving pre-ictal patient state."
            report['recommendation'] = "1. Maintain close nursing monitoring and place seizure precautions.\n2. Confirm lead integrity to rule out muscle/movement artifacts.\n3. Perform bedside neurological assessment of consciousness and motor function.\n4. Review serum antiepileptic drug levels and history."
    else:
        if probability < 0.20:
            report['status'] = 'NORMAL EEG PATTERN'
            report['status_color'] = HEX_NORMAL_GREEN
            report['risk_level'] = 'LOW RISK (Healthy Background)'
            report['summary'] = "Standard physiological background activity observed. Frequency power is distributed normally across typical waking frequency bands. No epileptiform discharges or rhythmic slow-waves were identified."
            report['recommendation'] = "1. Routine care; no specific neurological precautions required.\n2. Continue standard monitoring protocols.\n3. Document baseline frequency bands and patient state."
        else:
            report['status'] = 'BORDERLINE EEG PATTERN'
            report['status_color'] = HEX_ALERT_ORANGE
            report['risk_level'] = 'MILD RISK (Trace Abnormality)'
            report['summary'] = "EEG trace displays mild diffuse slowing or minor transient artifacts. No definitive epileptic seizure structures are present, but background complexity is slightly reduced."
            report['recommendation'] = "1. Correlate with physical state (drowsiness, medications).\n2. Repeat EEG if patient exhibits symptoms.\n3. Verify sensor impedance levels."
    dominant_band = max(rel_powers, key=rel_powers.get)
    report['dominant_band'] = dominant_band
    report['dominant_percentage'] = rel_powers[dominant_band] * 100
    band_details = []
    delta_p = rel_powers.get('Delta', 0) * 100
    delta_desc = f"**Delta ({delta_p:.1f}%)**: "
    delta_desc += "Significantly elevated. Indicates deep sleep or pathological slow-wave activity." if delta_p > 40 else "Normal level for a waking conscious adult."
    band_details.append(delta_desc)
    theta_p = rel_powers.get('Theta', 0) * 100
    theta_desc = f"**Theta ({theta_p:.1f}%)**: "
    theta_desc += "Elevated. Corresponds to drowsiness or pathological focal slowing." if theta_p > 30 else "Normal level representing standard relaxed background."
    band_details.append(theta_desc)
    alpha_p = rel_powers.get('Alpha', 0) * 100
    alpha_desc = f"**Alpha ({alpha_p:.1f}%)**: "
    alpha_desc += "Dominant. Indicates standard relaxed, awake state." if alpha_p > 35 else "Suppressed alpha background. Seizure onset often shows alpha desynchronization." if alpha_p < 10 and prediction == 1 else "Standard physiological baseline level."
    band_details.append(alpha_desc)
    beta_p = rel_powers.get('Beta', 0) * 100
    beta_desc = f"**Beta ({beta_p:.1f}%)**: "
    beta_desc += "Elevated. Reflects active concentration or medication influences." if beta_p > 30 else "Normal conscious cognitive background."
    band_details.append(beta_desc)
    gamma_p = rel_powers.get('Gamma', 0) * 100
    gamma_desc = f"**Gamma ({gamma_p:.1f}%)**: "
    gamma_desc += "Significantly elevated. Associated with active seizure onset." if gamma_p > 15 else "Standard low amplitude waking trace."
    band_details.append(gamma_desc)
    report['band_analyses'] = band_details
    return report