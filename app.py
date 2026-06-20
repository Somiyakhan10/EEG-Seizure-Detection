import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NeuroDetect - EEG Seizure Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed color scheme with NO WHITE
st.markdown("""
    <style>
        /* Main background - Dark Blue */
        .stApp {
            background-color: #0A1628;
        }
        
        /* Sidebar - Dark Blue Gradient */
        .css-1d391kg, .css-163i60w {
            background: linear-gradient(180deg, #0D1B2A 0%, #1B2D45 100%) !important;
            border-right: 1px solid #2A5CAA !important;
        }
        
        .css-1d391kg .css-1d391kg {
            background: transparent !important;
        }
        
        /* Sidebar text - Light Blue/Gray */
        .css-1d391kg .css-1d391kg, 
        .css-1d391kg p, 
        .css-1d391kg label,
        .css-1d391kg div,
        .css-1d391kg span {
            color: #B8D4E3 !important;
        }
        
        /* Sidebar headers - Bright Blue */
        .css-1d391kg h1, 
        .css-1d391kg h2, 
        .css-1d391kg h3 {
            color: #4A90D9 !important;
        }
        
        /* Sidebar radio buttons */
        .stRadio label {
            color: #B8D4E3 !important;
        }
        
        .stRadio div[role="radiogroup"] label {
            background-color: #1B2D45 !important;
            color: #B8D4E3 !important;
            border-radius: 5px;
            padding: 0.3rem 0.8rem;
            margin: 0.1rem 0;
        }
        
        .stRadio div[role="radiogroup"] label:hover {
            background-color: #2A5CAA !important;
            color: #FFFFFF !important;
        }
        
        .stRadio div[role="radiogroup"] label[data-checked="true"] {
            background-color: #2A5CAA !important;
            color: #FFFFFF !important;
            border: 1px solid #4A90D9 !important;
        }
        
        /* Main content headers */
        .main-header {
            color: #4A90D9;
            font-size: 2rem;
            font-weight: 700;
            padding: 1rem 0;
            border-bottom: 2px solid #2A5CAA;
            margin-bottom: 2rem;
        }
        
        /* Cards - Dark Blue */
        .metric-card {
            background: linear-gradient(135deg, #0D1B2A 0%, #1B2D45 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #2A5CAA;
            text-align: center;
            margin-bottom: 1rem;
            transition: transform 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(42, 92, 170, 0.3);
        }
        
        .metric-label {
            color: #7BB8E0;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        
        .metric-value.green { color: #4ADE80; }
        .metric-value.orange { color: #FBBF24; }
        .metric-value.red { color: #F87171; }
        .metric-value.blue { color: #4A90D9; }
        .metric-value.cyan { color: #22D3EE; }
        
        /* Report box */
        .report-box {
            background: linear-gradient(135deg, #0D1B2A 0%, #1B2D45 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #2A5CAA;
            margin-bottom: 1rem;
        }
        
        .report-box.critical { border-left-color: #F87171; }
        .report-box.warning { border-left-color: #FBBF24; }
        .report-box.success { border-left-color: #4ADE80; }
        
        .report-box h3 { color: #E2E8F0; }
        .report-box p { color: #B8D4E3; }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #2A5CAA 0%, #1A4A8A 100%);
            color: #E2E8F0 !important;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(42, 92, 170, 0.4);
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #3A6CBA 0%, #2A5CAA 100%) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1B2D45;
            color: #B8D4E3 !important;
            border-radius: 5px;
            padding: 0.5rem 1.5rem;
            border: 1px solid #2A5CAA;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #2A5CAA;
            color: #FFFFFF !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #2A5CAA;
            color: #FFFFFF !important;
            border: 1px solid #4A90D9;
        }
        
        /* Headers */
        h1 { color: #4A90D9 !important; }
        h2 { color: #22D3EE !important; }
        h3, h4 { color: #E2E8F0 !important; }
        p, li, span, div { color: #B8D4E3; }
        
        /* File uploader */
        .stFileUploader {
            background-color: #1B2D45;
            border: 2px dashed #2A5CAA;
            border-radius: 10px;
            padding: 2rem;
        }
        
        .stFileUploader label {
            color: #B8D4E3 !important;
        }
        
        /* Selectbox */
        .stSelectbox label {
            color: #B8D4E3 !important;
        }
        
        .stSelectbox div {
            color: #B8D4E3 !important;
        }
        
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1B2D45 !important;
            border-color: #2A5CAA !important;
        }
        
        /* Slider */
        .stSlider label {
            color: #B8D4E3 !important;
        }
        
        /* Radio buttons in main content */
        .stRadio label {
            color: #B8D4E3 !important;
        }
        
        /* Alerts */
        .stAlert {
            background-color: #1B2D45 !important;
            color: #B8D4E3 !important;
            border: 1px solid #2A5CAA !important;
        }
        
        .stAlert .stAlert-content {
            color: #B8D4E3 !important;
        }
        
        /* Sidebar metrics */
        .sidebar-metric {
            background-color: #1B2D45;
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 3px solid #2A5CAA;
            margin-bottom: 0.5rem;
        }
        
        .sidebar-metric-label {
            color: #7BB8E0;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        
        .sidebar-metric-value {
            color: #E2E8F0;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Dropdown options */
        .stSelectbox div[role="listbox"] {
            background-color: #1B2D45 !important;
            color: #B8D4E3 !important;
        }
        
        .stSelectbox div[role="option"] {
            background-color: #1B2D45 !important;
            color: #B8D4E3 !important;
        }
        
        .stSelectbox div[role="option"]:hover {
            background-color: #2A5CAA !important;
            color: #FFFFFF !important;
        }
        
        .stSelectbox div[role="option"][aria-selected="true"] {
            background-color: #2A5CAA !important;
            color: #FFFFFF !important;
        }
        
        /* Sidebar status */
        .sidebar-status {
            background-color: #1B2D45;
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 3px solid #4ADE80;
        }
        
        .sidebar-status.error {
            border-left-color: #F87171;
        }
        
        .sidebar-status-text {
            color: #4ADE80;
            margin: 0;
            font-size: 0.8rem;
        }
        
        .sidebar-status-text.error {
            color: #F87171;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_bundle():
    if os.path.exists("eeg_model.pkl"):
        return joblib.load("eeg_model.pkl")
    return None

model_bundle = load_model_bundle()

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #4A90D9; font-size: 1.8rem; margin: 0;">NeuroDetect</h1>
            <p style="color: #7BB8E0; font-size: 0.8rem; margin: 0;">EEG Seizure Detection Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "EEG Analysis", "Seizure Prediction", "Explainable AI", "Model Performance"],
        index=0
    )
    
    st.markdown("---")
    
    if model_bundle:
        st.markdown("""
            <div class="sidebar-status">
                <p class="sidebar-status-text">Model Loaded Successfully</p>
            </div>
        """, unsafe_allow_html=True)
        
        metrics = model_bundle.get('metrics', {})
        st.markdown(f"""
            <div style="margin-top: 0.5rem;">
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">Accuracy</div>
                    <div class="sidebar-metric-value">{metrics.get('accuracy', 0)*100:.1f}%</div>
                </div>
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">ROC-AUC Score</div>
                    <div class="sidebar-metric-value">{metrics.get('roc_auc', 0):.3f}</div>
                </div>
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">Sensitivity</div>
                    <div class="sidebar-metric-value">{metrics.get('sensitivity', 0)*100:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="sidebar-status error">
                <p class="sidebar-status-text error">Model Not Found</p>
            </div>
        """, unsafe_allow_html=True)
        st.info("Run: python model_training.py")
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.7rem;">
            Clinical Decision Support System
        </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE: HOME
# ============================================
if page == "Home":
    st.markdown("<h1 class='main-header'>EEG Seizure Detection Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #B8D4E3; font-size: 1.1rem;'>Welcome to NeuroDetect - an AI-powered clinical decision support system for EEG seizure detection.</p>", unsafe_allow_html=True)
    
    if model_bundle:
        metrics = model_bundle.get('metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value cyan">{metrics.get('accuracy', 0)*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sensitivity</div>
                    <div class="metric-value green">{metrics.get('sensitivity', 0)*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Specificity</div>
                    <div class="metric-value orange">{metrics.get('specificity', 0)*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ROC-AUC</div>
                    <div class="metric-value red">{metrics.get('roc_auc', 0):.3f}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2 style='color: #22D3EE;'>System Architecture</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="report-box">
                <h4>Signal Processing</h4>
                <p>Bandpass filtering isolates Delta, Theta, Alpha, Beta, and Gamma frequency bands</p>
            </div>
            <div class="report-box">
                <h4>Feature Extraction</h4>
                <p>Statistical, spectral, and wavelet features are computed from the EEG signal</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="report-box">
                <h4>Machine Learning</h4>
                <p>Gradient Boosting classifier predicts seizure probability from extracted features</p>
            </div>
            <div class="report-box">
                <h4>Explainable AI</h4>
                <p>SHAP values explain which features drive the prediction decisions</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE: EEG ANALYSIS
# ============================================
elif page == "EEG Analysis":
    st.markdown("<h1 class='main-header'>Interactive EEG Signal Analysis</h1>", unsafe_allow_html=True)
    
    if not model_bundle:
        st.warning("Please train the model first: python model_training.py")
    else:
        try:
            from utils import plot_raw_and_filtered_signals, plot_single_psd, plot_wavelet_scalogram
            from feature_engineering import FS, DEFAULT_BANDS
            
            test_data = model_bundle.get('test_data', {})
            raw_signals = np.array(test_data.get('raw_signals', []))
            labels = np.array(test_data.get('labels', []))
            
            if len(raw_signals) == 0:
                st.warning("No test data found. Please train the model again.")
            else:
                options = [f"Sample {i+1} (Seizure)" if labels[i]==1 else f"Sample {i+1} (Normal)" for i in range(len(raw_signals))]
                idx = st.selectbox("Select EEG Sample:", range(len(raw_signals)), format_func=lambda x: options[x])
                signal = raw_signals[idx]
                bands = model_bundle.get('bands', DEFAULT_BANDS)
                
                tab1, tab2, tab3 = st.tabs(["Waveform Decomposition", "Power Spectral Density", "Wavelet Scalogram"])
                with tab1:
                    fig = plot_raw_and_filtered_signals(signal, FS, bands)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    fig = plot_single_psd(signal, FS, bands)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with tab3:
                    fig = plot_wavelet_scalogram(signal, FS)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================
# PAGE: SEIZURE PREDICTION
# ============================================
elif page == "Seizure Prediction":
    st.markdown("<h1 class='main-header'>Real-Time Seizure Prediction</h1>", unsafe_allow_html=True)
    
    if not model_bundle:
        st.warning("Please train the model first: python model_training.py")
    else:
        try:
            from feature_engineering import extract_all_features_from_bands, calculate_relative_band_powers, FS, DEFAULT_BANDS
            from utils import get_clinical_interpretation, BAND_COLORS
            
            model = model_bundle.get('model')
            scaler = model_bundle.get('scaler')
            bands = model_bundle.get('bands', DEFAULT_BANDS)
            feature_names = model_bundle.get('feature_names', [])
            
            if model is None or scaler is None:
                st.error("Model or scaler not loaded properly. Please retrain the model.")
            else:
                st.markdown("### Select Input Source")
                source = st.radio("Input Source:", ["Synthetic Signal", "Upload CSV"], horizontal=True)
                
                input_signal = None
                if source == "Synthetic Signal":
                    col1, col2 = st.columns(2)
                    with col1:
                        signal_type = st.selectbox("Signal Type:", ["Normal", "Seizure"])
                    with col2:
                        noise = st.slider("Noise Level:", 0.1, 3.0, 0.5)
                    t = np.linspace(0, 1, int(FS))
                    if signal_type == "Seizure":
                        input_signal = 3 * (1.5 * np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 4 * t))
                        input_signal += np.random.normal(0, noise, len(t))
                    else:
                        input_signal = 0.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, noise, len(t))
                    st.success("Synthetic signal generated successfully")
                else:
                    uploaded = st.file_uploader("Upload CSV File (178 values)", type=['csv'])
                    if uploaded:
                        try:
                            data = pd.read_csv(uploaded, header=None)
                            vals = data.values.flatten()
                            if len(vals) == 178:
                                input_signal = np.array(vals)
                                st.success("File loaded successfully")
                            else:
                                st.error(f"Expected 178 values, got {len(vals)}")
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                
                if input_signal is not None:
                    with st.spinner("Analyzing EEG signal..."):
                        features = extract_all_features_from_bands(input_signal, bands, FS)
                        feat_df = pd.DataFrame([features])
                        for col in feature_names:
                            if col not in feat_df.columns:
                                feat_df[col] = 0.0
                        feat_df = feat_df[feature_names]
                        X_scaled = scaler.transform(feat_df)
                        pred = model.predict(X_scaled)[0]
                        prob = model.predict_proba(X_scaled)[0][1]
                        abs_p, rel_p = calculate_relative_band_powers(input_signal, FS, bands)
                        report = get_clinical_interpretation(pred, prob, rel_p)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        color = report['status_color']
                        status_class = "critical" if "CRITICAL" in report['status'] else "warning" if "WARNING" in report['status'] else "success"
                        st.markdown(f"""
                            <div class="report-box {status_class}">
                                <h3 style="color: {color};">{report['status']}</h3>
                                <p><b>Risk Level:</b> {report['risk_level']}</p>
                                <p>{report['summary']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob*100,
                            title={'text': "Seizure Probability", 'font': {'color': '#B8D4E3'}},
                            number={'suffix': "%", 'font': {'color': '#B8D4E3'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': '#B8D4E3'},
                                'bar': {'color': color},
                                'bgcolor': '#1B2D45',
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(74, 222, 128, 0.2)'},
                                    {'range': [30, 70], 'color': 'rgba(251, 191, 36, 0.2)'},
                                    {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': '#E2E8F0', 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=350, paper_bgcolor='#0D1B2A', font_color='#B8D4E3')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Frequency Band Analysis")
                    band_names = list(rel_p.keys())
                    percentages = [rel_p[b]*100 for b in band_names]
                    fig = go.Figure(go.Bar(
                        x=band_names,
                        y=percentages,
                        marker_color=[BAND_COLORS[b] for b in band_names],
                        text=[f"{p:.1f}%" for p in percentages],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=350,
                        paper_bgcolor='#0D1B2A',
                        plot_bgcolor='#1B2D45',
                        title="Relative Power by Frequency Band",
                        title_font_color='#B8D4E3',
                        xaxis_title="Frequency Band",
                        yaxis_title="Percentage of Total Power",
                        font_color='#B8D4E3'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.session_state['active_input_scaled'] = X_scaled[0]
                    st.session_state['active_input_pred'] = pred
                    st.session_state['active_input_prob'] = prob
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================
# PAGE: EXPLAINABLE AI
# ============================================
elif page == "Explainable AI":
    st.markdown("<h1 class='main-header'>Explainable AI and Interpretability</h1>", unsafe_allow_html=True)
    
    if not model_bundle:
        st.warning("Please train the model first: python model_training.py")
    else:
        try:
            from utils import plot_feature_importance, render_shap_summary_plot, render_shap_waterfall_plot
            import shap
            
            model = model_bundle.get('model')
            feature_names = model_bundle.get('feature_names', [])
            
            if model is None:
                st.error("Model not loaded properly. Please retrain.")
            else:
                tab1, tab2 = st.tabs(["Global Explanations", "Local Explanations"])
                
                with tab1:
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("### Feature Importance")
                        st.write("These features have the highest impact on the model's predictions.")
                        fig = plot_feature_importance(model.feature_importances_, feature_names, top_n=15)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### SHAP Global Summary")
                    st.write("Red indicates high feature values, blue indicates low feature values. Values to the right increase seizure risk.")
                    with st.spinner("Calculating SHAP values..."):
                        try:
                            explainer = shap.TreeExplainer(model)
                            test_data = model_bundle.get('test_data', {})
                            X_sample = np.array(test_data.get('scaled_features', []))[:60]
                            if len(X_sample) > 0:
                                fig = render_shap_summary_plot(explainer, X_sample, feature_names)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                            else:
                                st.warning("No test data available for SHAP analysis.")
                        except Exception as e:
                            st.warning(f"SHAP analysis limited: {str(e)}")
                
                with tab2:
                    if 'active_input_scaled' in st.session_state:
                        input_scaled = st.session_state['active_input_scaled']
                        pred = st.session_state['active_input_pred']
                        prob = st.session_state['active_input_prob']
                        
                        st.markdown(f"""
                            <div class="report-box">
                                <p><b>Prediction:</b> {'SEIZURE' if pred==1 else 'NON-SEIZURE'} | <b>Confidence:</b> {prob*100:.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner("Calculating SHAP explanation..."):
                            try:
                                explainer = shap.TreeExplainer(model)
                                sh_val = explainer.shap_values(np.array([input_scaled]))
                                if isinstance(sh_val, list):
                                    sh_val = sh_val[1]
                                base_val = explainer.expected_value
                                if isinstance(base_val, list):
                                    base_val = base_val[1]
                                fig = render_shap_waterfall_plot(explainer, base_val, sh_val[0], input_scaled, feature_names)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                            except Exception as e:
                                st.warning(f"Could not generate waterfall plot: {str(e)}")
                    else:
                        st.info("Go to Seizure Prediction page first to generate a prediction.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================
# PAGE: MODEL PERFORMANCE
# ============================================
elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance Metrics</h1>", unsafe_allow_html=True)
    
    if not model_bundle:
        st.warning("Please train the model first: python model_training.py")
    else:
        try:
            from utils import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
            
            metrics = model_bundle.get('metrics', {})
            test_data = model_bundle.get('test_data', {})
            y_test = np.array(test_data.get('labels', []))
            model = model_bundle.get('model')
            X_test = np.array(test_data.get('scaled_features', []))
            
            if len(y_test) == 0:
                st.warning("No test data available. Please retrain the model.")
            else:
                y_prob = model.predict_proba(X_test)[:, 1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Accuracy</div>
                            <div class="metric-value cyan">{metrics.get('accuracy', 0)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Sensitivity</div>
                            <div class="metric-value green">{metrics.get('sensitivity', 0)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Specificity</div>
                            <div class="metric-value orange">{metrics.get('specificity', 0)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">F1-Score</div>
                            <div class="metric-value blue">{metrics.get('f1_score', 0)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    cm = np.array(metrics.get('confusion_matrix', [[0,0],[0,0]]))
                    fig = plot_confusion_matrix(cm)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    curve = st.radio("Select Curve:", ["ROC Curve", "Precision-Recall Curve"], horizontal=True)
                    if curve == "ROC Curve":
                        fig = plot_roc_curve(y_test, y_prob)
                    else:
                        fig = plot_pr_curve(y_test, y_prob)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")