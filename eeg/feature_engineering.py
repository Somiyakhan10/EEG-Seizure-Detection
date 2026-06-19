import numpy as np
import pandas as pd
import pywt
from scipy import signal as scipy_signal
from scipy import stats
from scipy.stats import entropy as shannon_entropy

__all__ = [
    'DEFAULT_BANDS',
    'FS',
    'extract_all_features_from_bands',
    'apply_bandpass_filter',
    'calculate_relative_band_powers'
]

DEFAULT_BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 45.0)
}

FS = 178.0

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return scipy_signal.filtfilt(b, a, data)

def extract_statistical_features(signal_data):
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data)
    var_val = np.var(signal_data)
    skew_val = stats.skew(signal_data) if len(signal_data) > 2 and std_val > 1e-9 else 0.0
    kurt_val = stats.kurtosis(signal_data) if len(signal_data) > 3 and std_val > 1e-9 else 0.0
    rms_val = np.sqrt(np.mean(signal_data**2))
    ptp_val = np.ptp(signal_data)
    abs_signal = np.abs(signal_data) + 1e-10
    sh_entropy = shannon_entropy(abs_signal)
    
    features = {
        'mean': mean_val,
        'std': std_val,
        'variance': var_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'rms': rms_val,
        'peak_to_peak': ptp_val,
        'shannon_entropy': sh_entropy
    }
    return features

def extract_wavelet_features(signal_data, wavelet='db4', level=3):
    try:
        if len(signal_data) < 2**level:
            level = max(1, int(np.log2(len(signal_data))) - 1)
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        energies = [np.sum(np.array(coeff)**2) for coeff in coeffs]
        total_energy = sum(energies)
        energy_ratios = [e/total_energy for e in energies] if total_energy > 0 else energies
        wave_entropy = shannon_entropy(energy_ratios)
        features = {f'wavelet_energy_L{i}': energies[i] for i in range(len(energies))}
        features['wavelet_total_energy'] = total_energy
        features['wavelet_entropy'] = wave_entropy
    except Exception:
        fallback_energy = np.sum(signal_data**2)
        features = {
            'wavelet_energy_L0': fallback_energy,
            'wavelet_total_energy': fallback_energy,
            'wavelet_entropy': shannon_entropy(np.abs(signal_data) + 1e-10)
        }
    return features

def extract_spectral_features(signal_data, fs):
    nperseg = min(128, len(signal_data))
    freqs, psd = scipy_signal.welch(signal_data, fs, nperseg=nperseg)
    psd_sum = np.sum(psd)
    if psd_sum > 0:
        psd_norm = psd / psd_sum
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        spec_entropy /= np.log2(len(psd_norm))
    else:
        spec_entropy = 0.0
    features = {
        'spectral_entropy': spec_entropy,
        'total_spectral_power': psd_sum
    }
    return features, freqs, psd

def calculate_relative_band_powers(raw_signal, fs, bands_dict=DEFAULT_BANDS):
    if fs <= 0 or len(raw_signal) == 0:
        return {band: 0.0 for band in bands_dict}, {band: 0.0 for band in bands_dict}
    nperseg = min(128, len(raw_signal))
    freqs, psd = scipy_signal.welch(raw_signal, fs, nperseg=nperseg)
    absolute_powers = {}
    total_power = 0.0
    for band_name, (low, high) in bands_dict.items():
        band_idx = np.logical_and(freqs >= low, freqs <= high)
        if np.any(band_idx):
            band_power = np.sum(psd[band_idx])
        else:
            band_power = 0.0
        absolute_powers[band_name] = band_power
        total_power += band_power
    relative_powers = {}
    for band_name, power in absolute_powers.items():
        relative_powers[band_name] = power / total_power if total_power > 0 else 0.0
    return absolute_powers, relative_powers

def extract_all_features_from_bands(eeg_signal, bands_dict=DEFAULT_BANDS, fs=FS):
    all_features = {}
    raw_stat = extract_statistical_features(eeg_signal)
    raw_spec, freqs, psd = extract_spectral_features(eeg_signal, fs)
    for k, v in raw_stat.items():
        all_features[f'raw_{k}'] = v
    for k, v in raw_spec.items():
        all_features[f'raw_{k}'] = v
    abs_powers, rel_powers = calculate_relative_band_powers(eeg_signal, fs, bands_dict)
    for band_name, p_abs in abs_powers.items():
        all_features[f'{band_name}_abs_power'] = p_abs
        all_features[f'{band_name}_rel_power'] = rel_powers[band_name]
    for band_name, (low, high) in bands_dict.items():
        if high < fs/2:
            try:
                filtered = apply_bandpass_filter(eeg_signal, low, high, fs)
                stat_feats = extract_statistical_features(filtered)
                for k, v in stat_feats.items():
                    all_features[f'{band_name}_{k}'] = v
                wave_feats = extract_wavelet_features(filtered)
                for k, v in wave_feats.items():
                    all_features[f'{band_name}_{k}'] = v
                spec_feats, _, _ = extract_spectral_features(filtered, fs)
                for k, v in spec_feats.items():
                    all_features[f'{band_name}_{k}'] = v
            except Exception:
                continue
    return all_features

def prepare_feature_matrix(df_signals, y_labels=None, bands_dict=DEFAULT_BANDS, fs=FS):
    feature_list = []
    n_samples = len(df_signals)
    for i in range(n_samples):
        signal_data = df_signals.iloc[i].values
        feats = extract_all_features_from_bands(signal_data, bands_dict, fs)
        feature_list.append(feats)
    feature_df = pd.DataFrame(feature_list)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.mean())
    if y_labels is not None:
        feature_df['target'] = y_labels
    return feature_df