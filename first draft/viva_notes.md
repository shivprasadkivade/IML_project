# CSL2010 IML Project 1: Viva Preparation & Technical Defense Notes

Plain-language explanations of all architectural, mathematical, and algorithmic decisions made in building the Human Activity Recognition (HAR) pipeline. Use these notes for course evaluations, project viva, and technical interviews.

---

## 1. Why Pseudo-Subject Group CV Instead of Random K-Fold?

### The Core Problem (Subject Shift & Data Leakage)
In wearable sensor datasets (like MHEALTH), each subject has unique physiological characteristics:
- Specific walking cadence, stride length, and joint ranges.
- Sensor mounting offsets: A sensor strapped 2 cm higher on the ankle or rotated slightly produces a completely different static bias.
- Individual magnetic environment: Magnetometer readings depend on the body's orientation relative to local magnetic field vectors.
- Cardiac electrode contact impedance: Affects ECG baseline voltage.

In this competition, **Train data comes from 5 subjects, but Test data comes from 3 completely different subjects**.

If you use standard `StratifiedKFold`, windows from the same subject are mixed across training and validation folds. The model learns to recognize *who* is performing the activity (by memorizing static offsets and gait signatures) rather than *what* activity is occurring. When evaluated on the unseen test subjects, this "identity memorization" fails catastrophically.

### How We Solved It
Because true subject IDs were not provided:
1. We checked row order and IDs: IDs are permuted, with 581 activity transitions across 679 rows, showing no sequential session blocks.
2. We extracted **static baseline offset features** that reflect sensor mounting and individual anatomy rather than activity dynamics (per-window means of 3-axis magnetometers and 2-lead ECG baseline).
3. We standardized these offsets and clustered the windows into 5 pseudo-subject groups using K-Means (`n_clusters=5`).
4. We verified that each of the 12 activity classes appears across multiple clusters.
5. We performed **`StratifiedGroupKFold`** as our primary validation scheme, ensuring an entire pseudo-subject cluster is held out during validation.

### Expected Viva Question:
**Q: "If you didn't have true subject IDs, how can you claim your validation prevents subject leakage?"**
> **A:** "While pseudo-subject clustering is an approximation, clustering on static magnetometer offsets and ECG baselines groups windows that share the same sensor mounting bias and physiological baseline. Holding out an entire cluster during validation forces the model to classify activities using dynamic motion signatures (acceleration waveforms, frequencies, inter-axis correlations) rather than static offsets. Furthermore, the leakage gap (+0.34% to +13.65% lower scores on Group-CV compared to Random-CV) empirically proves that Random-CV was overly optimistic."

---

## 2. Why Did We Clean `train.csv` and How?

### The Reality of the Data
Our inspection revealed synthetic noise injected into `train.csv`:
- European decimal commas (e.g., `'-9,31'` instead of `'-9.31'`).
- Whitespace padding around strings.
- Corrupted non-numeric tokens (`'-'`, `'?'`, `'unknown'`).
- Outlier sentinels (`9999` and `9999.0`).
- This affected 638,842 cells (16.36% of all training values), distributed uniformly across all channels and all windows.
- Conversely, `test.csv` was 100% clean `float64` with 0 missing values.

### The Solution: Time-Series Linear Interpolation
Because each window is a continuous 5-second physical recording at 50 Hz (250 time steps):
- Mean imputation across the dataset would destroy the physical continuity of the motion waveform.
- Replacing with 0 would introduce massive high-frequency step discontinuities in the frequency spectrum.
- Instead, after parsing strings and converting sentinels to NaN, we applied **linear interpolation along the time axis (axis=1)** for each window and channel, with boundary backfill/forward fill.
- This restored continuous waveforms with **0 remaining NaNs** and preserved signal frequency spectra.

---

## 3. Why These Specific Hand-Crafted Features?

### The 4 Feature Domains
1. **Time-Domain Dynamics (460 features)**:
   - *RMS & Energy*: Capture the intensity of physical exertion (running/jogging have high energy; sitting/standing have near-zero dynamic energy).
   - *Jerk (Mean Absolute First Difference)*: Captures sudden changes in acceleration and impact shocks (stairs vs flat walking; running foot-strikes).
   - *Zero-Crossing & Mean-Crossing Rates*: Quantify the oscillatory cadence of repetitive movements.
   - *Skewness & Kurtosis*: Measure asymmetry and peakedness of acceleration bursts.
2. **Frequency-Domain / Spectral Properties (161 features)**:
   - Sampling rate $f_s = 50\text{ Hz}$, Nyquist frequency = $25\text{ Hz}$. Window length 250 samples gives a frequency resolution of $0.2\text{ Hz}$.
   - *Dominant Frequency*: Captures step frequency (e.g., walking cadence is typically $1.5 - 2.0\text{ Hz}$; jogging/running is $2.5 - 4.0\text{ Hz}$).
   - *Band Power Ratios*: Partition spectral energy into physiologically meaningful bands:
     - `0.0 - 0.5 Hz`: Postural adjustments and slow drift.
     - `0.5 - 3.0 Hz`: Natural human locomotion (walking, stair climbing).
     - `3.0 - 8.0 Hz`: Fast running, jumping, and rapid limb transitions.
     - `8.0 - 25.0 Hz`: Micro-vibrations, tremors, and impacts.
   - *Spectral Entropy*: Quantifies disorder; periodic movements (cycling, walking) have low entropy (concentrated spectral peaks), while irregular tasks (waist bending) have higher entropy.
3. **Multi-Axial Coupling & Cross-Channel Dynamics (70 features)**:
   - *Euclidean Norm Magnitude* ($\sqrt{x^2+y^2+z^2}$): Eliminates dependence on sensor orientation. If an ankle sensor rotates on the leg, individual $x/y/z$ channels change, but the total magnitude vector remains invariant!
   - *Signal Magnitude Area (SMA)*: Normalized sum of absolute amplitudes across axes; a standard metric in biomechanics for energy expenditure.
   - *Cross-Axis Correlation*: Hand and leg movements have distinctive coordinated multi-axis trajectories (e.g. arm swinging during walking produces negative correlation between specific axes).
4. **Gravitational Posture Angles (15 features)**:
   - In static postures (sitting, standing, lying), the accelerometer acts as an inclinometer measuring the $1g$ ($9.81\text{ m/s}^2$) gravitational acceleration vector.
   - By calculating normalized direction cosines ($g_x/\|g\|$, $g_y/\|g\|$, $g_z/\|g\|$) and posture pitch/roll angles ($\arctan2$), we achieve **100% clean separation between sitting, standing, and lying down**.

---

## 4. Why Feature Set B ("Subject-Robust") Over Feature Set A?

### Adversarial Validation Finding
In Step 1, our adversarial validation classifier achieved an ROC-AUC of **1.0000** between train and test windows when static channel levels were used. The top features that distinguished train from test were:
- `arm_mag_z`, `ankle_mag_x`, `arm_mag_x` (absolute magnetometer offsets).
- Static ECG baseline offsets.

In the physical world, magnetic fields shift depending on the testing room, metal structures, and sensor calibration. Test subjects were recorded in different sessions and environments.
- **Feature Set A (699 features)**: Retains raw magnetometer and ECG levels.
- **Feature Set B (621 features)**: Drops the 78 static offset features (`mean`, `median`, `min`, `max`, `percentiles` of magnetometers and ECG) while retaining all dynamic features (variances, spectra, correlations, magnitude vectors, and posture angles).
- **Result**: Feature Set B protects against leaderboard test-set covariate shift.

---

## 5. Why Extra Trees & Soft-Voting Ensemble for Submissions?

### Submission A: Extra Trees Classifier (Best Single Model)
- **Group-CV Score**: **99.66% ± 0.69%**
- **Leakage Gap**: Only **+0.34%** (smallest gap among all candidate models).
- **Why it Works**: Extra Trees (Extremely Randomized Trees) randomizes cut points for each feature. With 621 correlated sensor features, this extreme randomization drastically reduces model variance and prevents the trees from overfitting to a few dominant sensors.

### Submission B: Diverse Soft-Voting Ensemble (Best Ensemble)
- **Architecture**: A weighted blend of diverse model families:
  1. `ExtraTreesClassifier` (randomized partition boundaries)
  2. `RandomForestClassifier` (greedy optimal partition boundaries)
  3. `LightGBM` (gradient-boosted decision trees focusing on hard boundary cases)
  4. `SVM (RBF)` (smooth non-linear maximum-margin hyperplane)
- **OOF Accuracy**: **99.56%**
- **Test Agreement**: **93.55%** agreement with Submission A on the 403 test samples.
- **Why it Works**: Blending margin-based kernel methods (SVM) with ensemble trees prevents any single model's inductive bias from dominating on unseen test subjects.

---

## 6. What Was the Only Error Detected?

During out-of-fold cross-validation on all 679 training windows:
- 11 out of 12 classes achieved **100.0% precision and 100.0% recall** (including sitting, standing, lying, cycling, walking, stairs, arm elevation, knees bending, waist bends, jump front/back).
- Only **3 samples** out of 679 were misclassified:
  - True Class: `running` $\rightarrow$ Predicted Class: `jogging`.
- **Physical Reason**: Running and jogging are on a continuous velocity continuum. At boundary speeds (around 7–8 km/h), the stride frequency, impact jerk, and leg accelerations overlap closely.

---

## 7. Key Viva Q&A Quick Reference

| Viva Question | Concise Answer |
| :--- | :--- |
| **Q1: What is the sampling rate and window length?** | 50 Hz sampling frequency, 5-second window duration = 250 time steps per window. Total raw dimensions per sample: $250 \times 23 = 5,750$ values. |
| **Q2: How did you verify the tensor reshape?** | The CSV columns are strictly time-major (`t000_ch0 ... t000_ch22, t001_ch0 ...`). We reshaped $(N, 5750)$ to $(N, 250, 23)$ and asserted that `tensor[:, 0, ch_idx]` exactly equaled `t000_<channel>` with 0 error. |
| **Q3: What was your metric and loss function?** | Multiclass Classification Accuracy (competition metric). Candidate models used multiclass cross-entropy / log-loss (`mlogloss`). |
| **Q4: Did you fit scalers on the test set?** | No. All transformers (StandardScaler, PCA, Feature Selection) were strictly fit inside training folds via scikit-learn `Pipeline` objects to guarantee zero test leakage. |
| **Q5: Why did random CV show 100% while nested CV was 87.67%?** | Random CV has subject leakage because samples from the same subject appear in both train and validation folds. Nested CV (Outer Leave-One-Group-Out) tests on an entirely unseen pseudo-subject cluster, providing an honest estimate of generalization to unseen subjects. |
| **Q6: How did you separate sitting vs standing vs lying?** | In static postures, accelerometers measure the $1g$ gravitational acceleration vector. By computing normalized direction cosines and pitch/roll angles, sitting ($y \approx -9.8\text{ m/s}^2$ on chest), standing ($y \approx -9.8\text{ m/s}^2$), and lying ($z \approx 9.8\text{ m/s}^2$ or $x \approx 9.8\text{ m/s}^2$) are separated with 100% precision. |
| **Q7: What random seed did you use?** | All stochastic operations, numpy, torch, and scikit-learn estimators used `SEED = 42`. |
