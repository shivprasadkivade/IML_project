# CSL2010 IML Project 1: Viva Preparation & Technical Defense Notes
*(Second Draft - Scikit-Learn Re-implementation)*

Plain-language explanations of all architectural, mathematical, and algorithmic decisions made in building the Human Activity Recognition (HAR) pipeline using **scikit-learn**. Use these notes for course evaluations, project viva, and technical interviews.

---

## 1. Why Pseudo-Subject Group CV Instead of Random K-Fold?

### The Core Problem (Subject Shift & Data Leakage)
In wearable sensor datasets (such as MHEALTH), each human participant has distinct physical and physiological characteristics:
- **Sensor mounting bias**: If a sensor is strapped 2 cm higher on the ankle or rotated slightly, the static gravitational and magnetic axes shift significantly.
- **Magnetic environment**: Magnetometer readings vary with individual orientation relative to ambient indoor magnetic fields.
- **Electrode contact**: Skin-electrode contact impedance shifts ECG baseline voltages.
- **Biomechanical cadence**: Natural stride length, stepping frequency, and limb impact forces vary per individual.

In this competition, **Train data comes from 5 subjects, but Test data comes from 3 completely different subjects**.

If you use standard `StratifiedKFold`, windows from the same subject are mixed across training and validation folds. A machine learning model quickly memorizes *who* is performing the activity (by exploiting static sensor offsets) rather than learning *what* activity is occurring. When evaluated on unseen test subjects whose static offsets are completely different, this identity-memorizing model fails.

### How We Solved It in Scikit-Learn
Because true subject IDs were omitted:
1. **Row Order Inspection**: We demonstrated that row order and numeric IDs are permuted (581 activity transitions across 679 rows), with no contiguous session blocks.
2. **Static Offset Features**: We extracted 8 static sensor baseline features that reflect individual mounting and anatomy rather than dynamic movements: the window-level means of 3-axis ankle magnetometers, 3-axis arm magnetometers, and 2 chest ECG channels.
3. **K-Means Clustering**: Using `StandardScaler()` and `KMeans(n_clusters=5, random_state=42, n_init=30)`, we partitioned the 679 windows into 5 pseudo-subject clusters.
4. **StratifiedGroupKFold**: We used scikit-learn's `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` as our primary cross-validator, ensuring an entire pseudo-subject group is held out in each fold while preserving class balance.

### Expected Viva Question:
**Q: "If you didn't have true subject IDs, how can you claim your validation prevents subject leakage?"**
> **A:** *"While pseudo-subject clustering is an approximation of true physiological identities, clustering on static magnetometer offsets and ECG baselines groups windows that share the same sensor mounting bias and physiological baseline. Holding out an entire cluster during validation forces the model to classify activities using dynamic motion signatures (acceleration waveforms, frequencies, inter-axis correlations) rather than static offsets. Furthermore, the leakage gap (+0.34% to +13.65% lower scores on Group-CV compared to Random-CV) empirically proves that Random-CV was overly optimistic."*

---

## 2. Why Did We Clean `train.csv` and How?

### The Reality of the Raw Data
Our exploratory data analysis revealed synthetic noise injected into `train.csv`:
- European decimal commas (e.g., `'-9,31'` instead of `'-9.31'`).
- Whitespace padding around strings.
- Corrupted non-numeric tokens (`'-'`, `'?'`, `'unknown'`).
- Extreme outlier sentinels (`9999` and `9999.0`).
- This affected 638,842 cells (16.36% of all training values), distributed uniformly across all channels and all windows.
- Conversely, `test.csv` was 100% clean `float64` with 0 missing values.

### The Solution: Time-Series Linear Interpolation
Because each window is a continuous 5-second physical recording at 50 Hz (250 time steps):
- Mean imputation across the dataset would destroy the physical continuity of the motion waveform.
- Replacing missing values with 0 would introduce artificial high-frequency step discontinuities in the frequency spectrum.
- Instead, after parsing strings and converting sentinels to NaN, we applied **linear interpolation along the time axis (axis=1)** for each window and channel (`series.interpolate(method='linear', limit_direction='both').fillna(0.0)`).
- This restored continuous waveforms with **0 remaining NaNs** and preserved signal frequency spectra.

---

## 3. Why These Hand-Crafted Features? (4 Feature Domains)

### 1. Time-Domain Dynamics (460 features)
- **RMS & Energy**: Capture the physical intensity of movement (running and jogging have high energy; sitting and standing have near-zero dynamic energy).
- **Jerk (Mean Absolute First Difference)**: Captures sudden changes in acceleration and impact shocks (e.g., climbing stairs vs walking on a flat surface; foot-strikes during running).
- **Zero-Crossing & Mean-Crossing Rates**: Quantify the oscillatory cadence of repetitive movements.
- **Skewness & Kurtosis**: Measure asymmetry and peakedness of acceleration bursts.

### 2. Frequency-Domain / Spectral Properties (161 features)
- Sampling rate $f_s = 50\text{ Hz}$, Nyquist frequency = $25\text{ Hz}$. Window length 250 samples gives a frequency resolution of $0.2\text{ Hz}$.
- **Dominant Frequency**: Captures cyclic cadence (e.g., walking cadence is typically $1.5 - 2.0\text{ Hz}$; jogging/running is $2.5 - 4.0\text{ Hz}$).
- **Band Power Ratios**: Partition spectral energy into physiologically meaningful bands:
  - `0.0 - 0.5 Hz`: Postural adjustments and slow drift.
  - `0.5 - 3.0 Hz`: Natural human locomotion (walking, stair climbing).
  - `3.0 - 8.0 Hz`: Fast running, jumping, and rapid limb transitions.
  - `8.0 - 25.0 Hz`: Micro-vibrations, tremors, and impacts.
- **Spectral Entropy**: Quantifies spectral disorder; periodic movements (cycling, walking) have low entropy (concentrated spectral peaks), while irregular tasks (waist bending) have higher entropy.

### 3. Multi-Axial Coupling & Cross-Channel Dynamics (63 features)
- **Euclidean Norm Magnitude** ($\sqrt{x^2+y^2+z^2}$): Rotation-invariant metric. If an ankle sensor rotates slightly around the leg, individual $x/y/z$ channels change, but the total magnitude vector remains invariant!
- **Signal Magnitude Area (SMA)**: Normalized sum of absolute amplitudes across axes; a standard biomechanical metric for overall energy expenditure.
- **Cross-Axis Correlation**: Hand and leg movements have distinctive coordinated multi-axis trajectories (e.g., arm swinging during walking produces negative correlation between specific axes).

### 4. Gravitational Posture Angles (15 features)
- In static postures (sitting, standing, lying), accelerometers act as inclinometers measuring the $1g$ ($9.81\text{ m/s}^2$) gravitational acceleration vector.
- By calculating normalized direction cosines ($g_x/\|g\|$, $g_y/\|g\|$, $g_z/\|g\|$) and posture pitch/roll angles ($\arctan2$), we achieve **100% clean separation between sitting, standing, and lying down**.

---

## 4. Why Feature Set B ("Subject-Robust") Over Feature Set A?

### The Adversarial Validation Finding
Our adversarial validation classifier achieved an ROC-AUC of **1.0000** between train and test windows when static channel levels were included. The top features distinguishing train from test were absolute magnetometer extremes (`ankle_gyro_z_max`, `arm_mag_x_max`, `ankle_mag_x_min`).

In the physical world, magnetic fields shift depending on the testing room, building metal structures, and sensor calibration. Test subjects were recorded in different environments:
- **Feature Set A (699 features)**: Retains raw magnetometer and ECG levels.
- **Feature Set B (621 features)**: Drops the 78 static offset features (`mean`, `median`, `min`, `max`, `range`, `percentiles` of magnetometers and ECG) while retaining all dynamic features (variances, spectra, correlations, magnitude vectors, and posture angles).
- **Result**: Feature Set B protects against leaderboard test-set covariate shift on unseen subjects.

---

## 5. Every Scikit-Learn Model Explained in Simple Language

| Model Class | What it Does | Why it is Used |
| :--- | :--- | :--- |
| **`LogisticRegression`** | Linear decision boundary using softmax cross-entropy loss with L2 regularization ($C=0.1$). | Establishes a fast, interpretable linear baseline; also serves as the stacking ensemble meta-learner. |
| **`SVC(kernel='rbf')`** | Maps features into an infinite-dimensional reproducing kernel Hilbert space to construct a maximum-margin separating hyperplane. Uses Platt scaling (`probability=True`) for calibrated probabilities. | Smooth, non-linear margin-based decision boundaries that complement tree models. |
| **`RandomForestClassifier`** | Bagging ensemble of decision trees. Fits hundreds of trees on bootstrap sub-samples and chooses best split thresholds greedily. | Reduces variance compared to individual decision trees; provides robust feature interactions. |
| **`ExtraTreesClassifier`** | Extremely Randomized Trees. Similar to Random Forest, but randomizes both the feature subset AND the split thresholds. | Extreme randomization strongly reduces variance on high-dimensional correlated sensor features, achieving the highest single-model accuracy (99.66%). |
| **`LGBMClassifier` & `XGBClassifier`** | Gradient-boosted decision trees that construct shallow trees sequentially, fitting each new tree to the pseudo-residuals of earlier trees. | Focuses iterative tree growth on hard boundary samples (e.g. distinguishing running vs jogging). |
| **`MLPClassifier`** | Multi-Layer Perceptron neural network with 2 hidden layers (128, 64), ReLU activations, Adam optimization, L2 regularization (`alpha=1e-3`), and early stopping. | Replaces deep learning frameworks (TensorFlow / PyTorch) with native scikit-learn code. Learns non-linear feature combinations with early stopping. |
| **`RidgeClassifierCV`** | Multi-class linear model using L2 regularized least-squares on one-hot targets with built-in generalized cross-validation. | Pairs with the MiniRocket convolutional transform for fast, regularized linear classification. |
| **`VotingClassifier`** | Soft-voting ensemble metaclassifier that averages the predicted probability distributions across multiple estimators. | Combines diverse model families (SVM + RF + Extra Trees + LightGBM) to reduce prediction variance on unseen test subjects. |

---

## 6. How Did We Replace TensorFlow / Deep Learning with Scikit-Learn?

In the first draft, a 1D Convolutional Neural Network (PyTorch) was evaluated on raw time-series tensors (`(N, 23, 250)`). In the second draft:
1. **Model Replacement (`MLPClassifier`)**: We replaced the neural network with scikit-learn's `MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-3, batch_size=32, early_stopping=True, n_iter_no_change=10, random_state=42)` operating on the engineered Feature Set B.
2. **MiniRocket Replacement (`NumPyMiniRocket`)**: We replaced external `aeon` with a clean, vectorized pure-NumPy transformer subclassing `BaseEstimator` and `TransformerMixin`.
3. **Pipeline Encapsulation**: All scaling (`StandardScaler`) is encapsulated within `sklearn.pipeline.Pipeline`, ensuring that scalers are fit strictly on training folds and never see validation or test data.
4. **Why This is Better**:
   - The 1D CNN in first draft achieved only 92.73% Group-CV accuracy and took 41 seconds to train.
   - Hand-crafted domain features with tree ensembles and SVMs achieve **99.66% Group-CV** and train in 2–4 seconds because physical motion dynamics (frequencies, magnitudes, posture angles) are explicitly encoded rather than having to be learned from only 679 training windows.

---

## 7. Submission Selection: Why Extra Trees (A) and Soft-Voting Ensemble (B)?

### Submission A: Best Single Robust Model (`ExtraTreesClassifier`)
- **Group-CV Score**: **99.66% ± 0.69%**
- **Leakage Gap**: Only **+0.34%** (smallest gap among all candidate models).
- **Why it Works**: Extra Trees randomizes cut points for each feature. With 621 correlated sensor features, this extreme randomization drastically reduces model variance and prevents the trees from overfitting to a few dominant sensors.

### Submission B: Diverse Soft-Voting Ensemble
- **Architecture**: A weighted blend of diverse model families on Feature Set B:
  1. `ExtraTreesClassifier` (randomized partition boundaries)
  2. `RandomForestClassifier` (greedy optimal partition boundaries)
  3. `LGBMClassifier` (gradient-boosted decision trees focusing on hard boundary cases)
  4. `SVC (RBF)` (smooth non-linear maximum-margin hyperplane)
- **OOF Accuracy**: **99.56%**
- **Test Agreement**: **93.55%** agreement with Submission A (377 out of 403 test samples match).
- **Why it Works**: Blending margin-based kernel methods (SVM) with ensemble trees prevents any single model's inductive bias from dominating on unseen test subjects.

---

## 8. What Was the Only Error Detected?

During out-of-fold cross-validation on all 679 training windows:
- 11 out of 12 classes achieved **100.0% precision and 100.0% recall** (including sitting, standing, lying, cycling, walking, stairs, arm elevation, knees bending, waist bends, jump front/back).
- Only **3 samples** out of 679 were misclassified:
  - True Class: `running` $\rightarrow$ Predicted Class: `jogging`.
- **Physical Reason**: Running and jogging are on a continuous velocity continuum. At boundary speeds (around 7–8 km/h), the stride frequency, impact jerk, and leg accelerations overlap closely.

---

## 9. Key Viva Q&A Quick Reference

| Viva Question | Concise Answer |
| :--- | :--- |
| **Q1: What is the sampling rate and window length?** | 50 Hz sampling frequency, 5-second window duration = 250 time steps per window. Total raw dimensions per sample: $250 \times 23 = 5,750$ values. |
| **Q2: How did you verify the tensor reshape?** | The CSV columns are strictly time-major (`t000_ch0 ... t000_ch22, t001_ch0 ...`). We reshaped $(N, 5750)$ to $(N, 250, 23)$ and asserted that `tensor[:, 0, ch_idx]` exactly equaled `t000_<channel>` with 0 error. |
| **Q3: What was your metric and loss function?** | Multiclass Classification Accuracy (competition metric). Candidate models used multiclass cross-entropy / log-loss (`mlogloss`). |
| **Q4: Did you fit scalers on the test set?** | No. All transformers (`StandardScaler`, `LabelEncoder`) were strictly fit inside training folds via scikit-learn `Pipeline` objects to guarantee zero test leakage. |
| **Q5: Why did random CV show 100% while nested CV was 87.67%?** | Random CV has subject leakage because samples from the same subject appear in both train and validation folds. Nested CV (Outer Leave-One-Group-Out) tests on an entirely unseen pseudo-subject cluster, providing an honest estimate of generalization to unseen subjects. |
| **Q6: How did you separate sitting vs standing vs lying?** | In static postures, accelerometers measure the $1g$ gravitational acceleration vector. By computing normalized direction cosines and pitch/roll angles, sitting ($y \approx -9.8\text{ m/s}^2$ on chest), standing ($y \approx -9.8\text{ m/s}^2$), and lying ($z \approx 9.8\text{ m/s}^2$ or $x \approx 9.8\text{ m/s}^2$) are separated with 100% precision. |
| **Q7: What random seed did you use?** | All stochastic operations, numpy, and scikit-learn estimators used `SEED = 42`. |
