"""
CSL2010 IML Project 1: Feature Engineering (features.py)
File: features.py (Second Draft - Scikit-Learn Re-implementation)

Purpose:
Extracts domain-informed features from (N, 250, 23) wearable sensor window tensors:
1. Time-Domain Features (20 features x 23 channels = 460 features).
2. Frequency-Domain Features via Real FFT (7 features x 23 channels = 161 features).
3. Cross-Channel Multi-Axial Dynamics (7 sensor triplets x 9 features = 63 features).
4. Gravitational Posture Angles (3 accelerometer triplets x 5 features = 15 features).
Total Feature Set A: 699 features.
Feature Set B (Subject-Robust): 621 features (drops 78 static offset features).

Evaluates Feature Sets A and B using leak-free scikit-learn Pipelines:
- StandardScaler: Fits only on training folds to prevent test set data leakage.
- LogisticRegression: Linear model with L2 regularization.
- RandomForestClassifier: Non-linear tree ensemble.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# Scikit-learn imports:
# - Pipeline: Chains data transformation (StandardScaler) with an estimator to guarantee no data leakage during CV.
# - StandardScaler: Standardizes features by centering to mean 0 and scaling to unit variance.
# - LogisticRegression: Linear classification model using cross-entropy loss.
# - RandomForestClassifier: Ensemble classifier that fits multiple decision trees on sub-samples.
# - accuracy_score: Computes classification accuracy (fraction of correct predictions).
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Resolve directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Sampling frequency and window length
FS = 50.0        # 50 Hz sampling rate
WINDOW_LEN = 250 # 5 seconds * 50 Hz = 250 samples

# Triaxial sensor channel triplets
TRI_AXIAL_SENSORS = {
    'chest_acc': ['chest_acc_x', 'chest_acc_y', 'chest_acc_z'],
    'ankle_acc': ['ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z'],
    'ankle_gyro': ['ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'],
    'ankle_mag': ['ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z'],
    'arm_acc': ['arm_acc_x', 'arm_acc_y', 'arm_acc_z'],
    'arm_gyro': ['arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z'],
    'arm_mag': ['arm_mag_x', 'arm_mag_y', 'arm_mag_z']
}

ACCEL_SENSORS = ['chest_acc', 'ankle_acc', 'arm_acc']


def extract_channel_time_features(sig):
    """
    Computes 20 time-domain statistical and dynamical features for a 1D signal of length 250.
    """
    mean_val = np.mean(sig)
    std_val = np.std(sig)
    min_val = np.min(sig)
    max_val = np.max(sig)
    range_val = max_val - min_val
    median_val = np.median(sig)

    p10 = np.percentile(sig, 10)
    p25 = np.percentile(sig, 25)
    p75 = np.percentile(sig, 75)
    p90 = np.percentile(sig, 90)
    iqr_val = p75 - p25

    # Skewness (asymmetry) and Kurtosis (tailedness/burstiness)
    skew_val = float(stats.skew(sig, bias=False)) if std_val > 1e-6 else 0.0
    kurt_val = float(stats.kurtosis(sig, bias=False)) if std_val > 1e-6 else 0.0

    # RMS and Energy: capture overall exertion intensity
    rms_val = np.sqrt(np.mean(sig ** 2))
    energy_val = np.sum(sig ** 2) / len(sig)

    # Mean absolute deviation (MAD)
    mad_val = np.mean(np.abs(sig - mean_val))

    # Crossing rates: cadence and oscillation frequency
    centered = sig - mean_val
    mean_cross_rate = np.sum(np.diff(centered > 0).astype(bool)) / len(sig)
    zero_cross_rate = np.sum(np.diff(sig > 0).astype(bool)) / len(sig)

    # Jerk: mean absolute first difference (sample-to-sample impact/shock)
    jerk_val = np.mean(np.abs(np.diff(sig)))

    # Peak count: counting local maxima
    peaks, _ = find_peaks(sig)
    peak_count = len(peaks)

    return [
        mean_val, std_val, min_val, max_val, range_val, median_val,
        p10, p25, p75, p90, iqr_val, skew_val, kurt_val,
        rms_val, energy_val, mad_val, mean_cross_rate, zero_cross_rate,
        jerk_val, peak_count
    ]


TIME_FEATURE_NAMES = [
    'mean', 'std', 'min', 'max', 'range', 'median',
    'p10', 'p25', 'p75', 'p90', 'iqr', 'skew', 'kurtosis',
    'rms', 'energy', 'mad', 'mean_cross_rate', 'zero_cross_rate',
    'jerk', 'peak_count'
]


def extract_channel_freq_features(sig, fs=FS):
    """
    Computes 7 frequency-domain spectral features via Real FFT for a 1D signal of length 250.
    """
    n = len(sig)
    # Detrend / center signal prior to FFT
    sig_centered = sig - np.mean(sig)

    # Real Fast Fourier Transform
    fft_vals = np.fft.rfft(sig_centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(fft_vals) ** 2) / n
    total_psd = np.sum(psd) + 1e-12

    # Dominant frequency (exclude DC component at index 0)
    if len(psd) > 1:
        dom_idx = np.argmax(psd[1:]) + 1
        dom_freq = freqs[dom_idx]
    else:
        dom_freq = 0.0

    # Physiologically meaningful band power ratios:
    # 0.0 - 0.5 Hz: Postural drift
    # 0.5 - 3.0 Hz: Locomotion / walking / stepping
    # 3.0 - 8.0 Hz: Fast transitions / running / jumping
    # 8.0 - 25.0 Hz: High-frequency impacts and vibrations
    band1 = np.sum(psd[(freqs >= 0.0) & (freqs < 0.5)]) / total_psd
    band2 = np.sum(psd[(freqs >= 0.5) & (freqs < 3.0)]) / total_psd
    band3 = np.sum(psd[(freqs >= 3.0) & (freqs < 8.0)]) / total_psd
    band4 = np.sum(psd[(freqs >= 8.0) & (freqs <= 25.0)]) / total_psd

    # Spectral entropy: measure of spectral flatness/disorder
    psd_norm = psd / total_psd
    psd_norm_nz = psd_norm[psd_norm > 1e-12]
    spectral_entropy = -np.sum(psd_norm_nz * np.log2(psd_norm_nz))

    # Spectral centroid: center frequency of the power distribution
    spectral_centroid = np.sum(freqs * psd) / total_psd

    return [
        dom_freq, band1, band2, band3, band4,
        spectral_entropy, spectral_centroid
    ]


FREQ_FEATURE_NAMES = [
    'dom_freq', 'band_0_05_ratio', 'band_05_3_ratio',
    'band_3_8_ratio', 'band_8_25_ratio', 'spectral_entropy', 'spectral_centroid'
]


def extract_cross_channel_features(window_tensor, channel_map):
    """
    Computes 3-axis magnitude stats, SMA, and cross-axis correlations for all 7 tri-axial sensors,
    plus gravitational orientation features for the 3 accelerometers.
    """
    features = []
    names = []

    for s_name, axes in TRI_AXIAL_SENSORS.items():
        idx_x, idx_y, idx_z = [channel_map[ax] for ax in axes]
        x, y, z = window_tensor[:, idx_x], window_tensor[:, idx_y], window_tensor[:, idx_z]

        # Euclidean norm magnitude: rotation-invariant physical acceleration/rotation vector
        mag = np.sqrt(x**2 + y**2 + z**2)
        features.extend([
            np.mean(mag), np.std(mag), np.max(mag), np.min(mag),
            np.sum(mag**2) / len(mag)  # Magnitude energy
        ])
        names.extend([
            f"{s_name}_mag_mean", f"{s_name}_mag_std",
            f"{s_name}_mag_max", f"{s_name}_mag_min",
            f"{s_name}_mag_energy"
        ])

        # Signal Magnitude Area (SMA)
        sma = np.mean(np.abs(x) + np.abs(y) + np.abs(z))
        features.append(sma)
        names.append(f"{s_name}_sma")

        # Inter-axis Pearson correlation coefficients
        std_x, std_y, std_z = np.std(x), np.std(y), np.std(z)
        corr_xy = np.corrcoef(x, y)[0, 1] if std_x > 1e-6 and std_y > 1e-6 else 0.0
        corr_xz = np.corrcoef(x, z)[0, 1] if std_x > 1e-6 and std_z > 1e-6 else 0.0
        corr_yz = np.corrcoef(y, z)[0, 1] if std_y > 1e-6 and std_z > 1e-6 else 0.0

        features.extend([corr_xy, corr_xz, corr_yz])
        names.extend([f"{s_name}_corr_xy", f"{s_name}_corr_xz", f"{s_name}_corr_yz"])

    # Gravitational posture features from low-passed accelerometer signals
    for a_name in ACCEL_SENSORS:
        axes = TRI_AXIAL_SENSORS[a_name]
        idx_x, idx_y, idx_z = [channel_map[ax] for ax in axes]
        mean_x = np.mean(window_tensor[:, idx_x])
        mean_y = np.mean(window_tensor[:, idx_y])
        mean_z = np.mean(window_tensor[:, idx_z])

        g_norm = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2) + 1e-6
        # Normalized direction cosines of the 1g gravity vector
        gx_norm, gy_norm, gz_norm = mean_x / g_norm, mean_y / g_norm, mean_z / g_norm

        # Posture pitch and roll angles
        pitch = np.arctan2(mean_x, np.sqrt(mean_y**2 + mean_z**2) + 1e-6)
        roll = np.arctan2(mean_y, np.sqrt(mean_x**2 + mean_z**2) + 1e-6)

        features.extend([gx_norm, gy_norm, gz_norm, pitch, roll])
        names.extend([
            f"{a_name}_gx_norm", f"{a_name}_gy_norm", f"{a_name}_gz_norm",
            f"{a_name}_pitch", f"{a_name}_roll"
        ])

    return features, names


def extract_all_features_from_tensor(tensor, channels):
    """
    Extracts complete tabular feature matrix from a 3D sensor tensor of shape (N, 250, 23).
    Returns a pandas DataFrame of shape (N, 699).
    """
    n_samples, n_timesteps, n_channels = tensor.shape
    channel_map = {ch: idx for idx, ch in enumerate(channels)}

    rows = []
    feature_names = None

    for i in range(n_samples):
        sample_features = []
        sample_names = []

        # 1. Per-channel time and frequency features
        for c_idx, ch in enumerate(channels):
            sig = tensor[i, :, c_idx]

            # Time features (20)
            t_feats = extract_channel_time_features(sig)
            sample_features.extend(t_feats)
            if feature_names is None:
                sample_names.extend([f"{ch}_{feat}" for feat in TIME_FEATURE_NAMES])

            # Frequency features (7)
            f_feats = extract_channel_freq_features(sig, fs=FS)
            sample_features.extend(f_feats)
            if feature_names is None:
                sample_names.extend([f"{ch}_{feat}" for feat in FREQ_FEATURE_NAMES])

        # 2. Cross-channel and posture features (78)
        cross_feats, cross_names = extract_cross_channel_features(tensor[i], channel_map)
        sample_features.extend(cross_feats)
        if feature_names is None:
            sample_names.extend(cross_names)
            feature_names = sample_names

        rows.append(sample_features)

    df_features = pd.DataFrame(rows, columns=feature_names)

    # Safety sanitization: replace any non-finite entries with column medians
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median().fillna(0.0))

    return df_features


def get_subject_robust_feature_names(all_feature_names):
    """
    Builds Feature Set B (Subject-Robust):
    Drops static baseline offset features that shift across subjects:
    - Magnetometer raw levels: means, medians, mins, maxs, ranges, percentiles of ankle_mag and arm_mag.
    - ECG baseline levels: means, medians, mins, maxs, ranges, percentiles of chest_ecg.
    Retains dynamic properties: std, IQR, MAD, RMS, jerk, crossing rates,
    FFT spectral bands, entropy, cross-axis correlations, magnitude dynamics, posture angles.
    """
    drop_prefixes = ['ankle_mag_', 'arm_mag_', 'chest_ecg_']
    drop_stats = ['_mean', '_median', '_min', '_max', '_p10', '_p25', '_p75', '_p90', '_range']

    robust_features = []
    for f in all_feature_names:
        is_sensitive = False
        for prefix in drop_prefixes:
            if f.startswith(prefix):
                for stat in drop_stats:
                    if f.endswith(stat):
                        is_sensitive = True
                        break
            if is_sensitive:
                break
        if not is_sensitive:
            robust_features.append(f)

    return robust_features


def main():
    print("=" * 80)
    print("STEP 3: FEATURE ENGINEERING (SCIKIT-LEARN SECOND DRAFT)")
    print("=" * 80)

    from cv import load_and_preprocess_train, get_group_cv

    # 1. Load preprocessed tensors
    print("\n[1] Loading preprocessed train and test data...")
    train_path = os.path.join(BASE_DIR, 'train.csv')
    test_path = os.path.join(BASE_DIR, 'test.csv')

    train_ids, train_targets, train_tensor, channels = load_and_preprocess_train(train_path)

    test_df = pd.read_csv(test_path, low_memory=False)
    sig_cols_test = [c for c in test_df.columns if c != 'id']
    test_tensor = test_df[sig_cols_test].values.reshape(-1, 250, 23)

    print(f"  Train tensor: {train_tensor.shape}, Test tensor: {test_tensor.shape}")

    # 2. Extract features
    print("\n[2] Extracting comprehensive features from train tensor...")
    df_feat_train = extract_all_features_from_tensor(train_tensor, channels)
    print(f"  Extracted Train Features: {df_feat_train.shape[1]} features across {df_feat_train.shape[0]} windows.")

    print("\n[3] Extracting comprehensive features from test tensor...")
    df_feat_test = extract_all_features_from_tensor(test_tensor, channels)
    print(f"  Extracted Test Features: {df_feat_test.shape[1]} features across {df_feat_test.shape[0]} windows.")

    # Validation: zero NaNs or Infs
    assert df_feat_train.isna().sum().sum() == 0, "NaN values detected in train features!"
    assert df_feat_test.isna().sum().sum() == 0, "NaN values detected in test features!"
    print("  Verification passed: Exactly 0 NaN or Inf values in extracted features.")

    # 3. Define Feature Sets A and B
    all_features = df_feat_train.columns.tolist()
    robust_features = get_subject_robust_feature_names(all_features)
    print(f"\n[4] Feature Set Definitions:")
    print(f"  Feature Set A (All Features):            {len(all_features)} features")
    print(f"  Feature Set B (Subject-Robust Dynamics): {len(robust_features)} features (dropped {len(all_features) - len(robust_features)} static offset features)")

    # 4. Save feature sets for fast pipeline training
    features_train_path = os.path.join(OUTPUTS_DIR, 'features_train.csv')
    features_test_path = os.path.join(OUTPUTS_DIR, 'features_test.csv')
    df_feat_train.to_csv(features_train_path, index=False)
    df_feat_test.to_csv(features_test_path, index=False)
    print(f"  Saved feature caches to {features_train_path} and {features_test_path}")

    # 5. Compare Feature Sets A and B using StratifiedGroupKFold CV in scikit-learn Pipelines
    print("\n[5] Comparing Feature Set A vs Feature Set B using StratifiedGroupKFold...")
    pseudo_groups_path = os.path.join(OUTPUTS_DIR, 'pseudo_groups.csv')
    pseudo_groups = pd.read_csv(pseudo_groups_path)['pseudo_group'].values
    group_cv = get_group_cv(n_splits=5, seed=42)

    def evaluate_feature_set(feat_cols, set_name):
        X = df_feat_train[feat_cols].values
        y = train_targets

        # Scikit-learn Pipelines guarantee that StandardScaler is fit ONLY on training folds
        pipe_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, max_iter=500, random_state=42))
        ])
        pipe_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
        ])

        lr_scores, rf_scores = [], []
        for tr_idx, val_idx in group_cv.split(X, y, groups=pseudo_groups):
            pipe_lr.fit(X[tr_idx], y[tr_idx])
            pipe_rf.fit(X[tr_idx], y[tr_idx])
            lr_scores.append(accuracy_score(y[val_idx], pipe_lr.predict(X[val_idx])))
            rf_scores.append(accuracy_score(y[val_idx], pipe_rf.predict(X[val_idx])))

        print(f"  {set_name} ({len(feat_cols)} features):")
        print(f"    Logistic Regression (C=0.1) Group-CV: {np.mean(lr_scores)*100:.2f}% +/- {np.std(lr_scores)*100:.2f}%")
        print(f"    Random Forest       (depth=8) Group-CV: {np.mean(rf_scores)*100:.2f}% +/- {np.std(rf_scores)*100:.2f}%")
        return {
            'lr_mean': np.mean(lr_scores), 'lr_std': np.std(lr_scores),
            'rf_mean': np.mean(rf_scores), 'rf_std': np.std(rf_scores)
        }

    res_A = evaluate_feature_set(all_features, "Feature Set A (All)")
    res_B = evaluate_feature_set(robust_features, "Feature Set B (Subject-Robust)")

    # 6. Save Markdown Summary Deliverable
    feat_summary_path = os.path.join(OUTPUTS_DIR, 'features_summary.md')
    with open(feat_summary_path, 'w') as f:
        f.write("# Feature Engineering Summary (STEP 3)\n\n")
        f.write("## 1. Domain-Informed Feature Hierarchy\n")
        f.write("From the 5-second 50 Hz windows `(N, 250, 23)`, we extracted features across 4 distinct domains:\n\n")
        f.write("1. **Time Domain (20 features per channel x 23 channels = 460 features)**:\n")
        f.write("   - Basic stats: Mean, std, min, max, range, median, IQR, percentiles (10, 25, 75, 90).\n")
        f.write("   - Shape & Dispersion: Skewness, kurtosis, MAD (mean absolute deviation).\n")
        f.write("   - Dynamics & Energy: RMS, signal energy, jerk (mean absolute first difference), mean-crossing rate, zero-crossing rate, peak count.\n")
        f.write("2. **Frequency Domain (7 features per channel x 23 channels = 161 features)**:\n")
        f.write("   - Dominant frequency (peak spectral component).\n")
        f.write("   - Power Spectral Density (PSD) band power ratios: `[0-0.5 Hz]` (postural drift), `[0.5-3.0 Hz]` (locomotion/stepping), `[3.0-8.0 Hz]` (vibrations/rapid transitions), `[8.0-25.0 Hz]` (high-frequency impacts).\n")
        f.write("   - Spectral entropy & spectral centroid.\n")
        f.write("3. **Cross-Channel & Multi-Axial Coupling (7 sensors x 9 features = 63 features)**:\n")
        f.write("   - Multi-axis magnitude dynamics: mean, std, min, max, energy of Euclidean norm $\\sqrt{x^2+y^2+z^2}$.\n")
        f.write("   - Signal Magnitude Area (SMA): $\\frac{1}{T}\\sum (|x| + |y| + |z|)$.\n")
        f.write("   - Inter-axis Pearson correlation: $(x,y)$, $(x,z)$, $(y,z)$.\n")
        f.write("4. **Posture & Gravitational Orientation (3 accelerometers x 5 features = 15 features)**:\n")
        f.write("   - Normalized gravity direction cosines ($g_x/\\|g\\|$, $g_y/\\|g\\|$, $g_z/\\|g\\|$).\n")
        f.write("   - Pitch angle $\\arctan2(x, \\sqrt{y^2+z^2})$ and roll angle $\\arctan2(y, \\sqrt{x^2+z^2})$ to robustly discriminate sitting, standing, and lying down.\n\n")
        f.write("## 2. Feature Set Comparison (Group-CV Performance)\n\n")
        f.write("| Feature Set | Num Features | Logistic Reg CV Acc (%) | Random Forest CV Acc (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Set A (All Features)** | {len(all_features)} | {res_A['lr_mean']*100:.2f}% ± {res_A['lr_std']*100:.2f}% | {res_A['rf_mean']*100:.2f}% ± {res_A['rf_std']*100:.2f}% |\n")
        f.write(f"| **Set B (Subject-Robust)** | {len(robust_features)} | {res_B['lr_mean']*100:.2f}% ± {res_B['lr_std']*100:.2f}% | {res_B['rf_mean']*100:.2f}% ± {res_B['rf_std']*100:.2f}% |\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> Feature Set B removes static offsets (raw magnetometer and ECG levels) that shift across unseen test subjects while retaining all underlying dynamic signatures.\n")

    print(f"\nFeature engineering complete! Summary saved to {feat_summary_path}")


if __name__ == '__main__':
    main()
