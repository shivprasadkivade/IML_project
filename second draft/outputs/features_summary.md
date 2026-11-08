# Feature Engineering Summary (STEP 3)

## 1. Domain-Informed Feature Hierarchy
From the 5-second 50 Hz windows `(N, 250, 23)`, we extracted features across 4 distinct domains:

1. **Time Domain (20 features per channel x 23 channels = 460 features)**:
   - Basic stats: Mean, std, min, max, range, median, IQR, percentiles (10, 25, 75, 90).
   - Shape & Dispersion: Skewness, kurtosis, MAD (mean absolute deviation).
   - Dynamics & Energy: RMS, signal energy, jerk (mean absolute first difference), mean-crossing rate, zero-crossing rate, peak count.
2. **Frequency Domain (7 features per channel x 23 channels = 161 features)**:
   - Dominant frequency (peak spectral component).
   - Power Spectral Density (PSD) band power ratios: `[0-0.5 Hz]` (postural drift), `[0.5-3.0 Hz]` (locomotion/stepping), `[3.0-8.0 Hz]` (vibrations/rapid transitions), `[8.0-25.0 Hz]` (high-frequency impacts).
   - Spectral entropy & spectral centroid.
3. **Cross-Channel & Multi-Axial Coupling (7 sensors x 9 features = 63 features)**:
   - Multi-axis magnitude dynamics: mean, std, min, max, energy of Euclidean norm $\sqrt{x^2+y^2+z^2}$.
   - Signal Magnitude Area (SMA): $\frac{1}{T}\sum (|x| + |y| + |z|)$.
   - Inter-axis Pearson correlation: $(x,y)$, $(x,z)$, $(y,z)$.
4. **Posture & Gravitational Orientation (3 accelerometers x 5 features = 15 features)**:
   - Normalized gravity direction cosines ($g_x/\|g\|$, $g_y/\|g\|$, $g_z/\|g\|$).
   - Pitch angle $\arctan2(x, \sqrt{y^2+z^2})$ and roll angle $\arctan2(y, \sqrt{x^2+z^2})$ to robustly discriminate sitting, standing, and lying down.

## 2. Feature Set Comparison (Group-CV Performance)

| Feature Set | Num Features | Logistic Reg CV Acc (%) | Random Forest CV Acc (%) |
| :--- | :---: | :---: | :---: |
| **Set A (All Features)** | 699 | 97.08% ± 4.61% | 99.83% ± 0.34% |
| **Set B (Subject-Robust)** | 621 | 95.41% ± 7.27% | 98.79% ± 2.41% |

> [!IMPORTANT]
> Feature Set B removes static offsets (raw magnetometer and ECG levels) that shift across unseen test subjects while retaining all underlying dynamic signatures.
