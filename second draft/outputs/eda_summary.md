# Exploratory Data Analysis (EDA) Summary: CSL2010 IML Project 1

## 1. Dataset Dimensions and Column Schema
- **Train Samples**: 679 windows (5-second duration at 50 Hz = 250 time steps per window)
- **Test Samples**: 403 windows
- **Total Signal Columns**: 5750 (exactly 250 time steps x 23 channels)
- **Column Naming & Order Convention**: Strictly time-major formatted as `t<000-249>_<channel>`.
- **Reshape Verification**: Reshaping `(N, 5750)` to `(N, 250, 23)` verified with 0 error; `t000_<ch>` maps exactly to index `[:, 0, ch_idx]`.
- **Sensors & Channels (23 total)**:
  - Chest (5 channels): `chest_acc_x`, `chest_acc_y`, `chest_acc_z`, `chest_ecg_1`, `chest_ecg_2`
  - Ankle (9 channels): `ankle_acc_x`, `ankle_acc_y`, `ankle_acc_z`, `ankle_gyro_x`, `ankle_gyro_y`, `ankle_gyro_z`, `ankle_mag_x`, `ankle_mag_y`, `ankle_mag_z`
  - Arm (9 channels): `arm_acc_x`, `arm_acc_y`, `arm_acc_z`, `arm_gyro_x`, `arm_gyro_y`, `arm_gyro_z`, `arm_mag_x`, `arm_mag_y`, `arm_mag_z`

## 2. Target Class Distribution & Imbalance
The dataset spans 12 physical activities. Class counts and proportions:

| Class Label | Window Count | Percentage (%) |
| :--- | :---: | :---: |
| `cycling` | 67.0 | 9.87% |
| `jogging` | 66.0 | 9.72% |
| `lying` | 66.0 | 9.72% |
| `walking` | 63.0 | 9.28% |
| `running` | 61.0 | 8.98% |
| `standing` | 59.0 | 8.69% |
| `sitting` | 59.0 | 8.69% |
| `climbing_stairs` | 56.0 | 8.25% |
| `knees_bending` | 55.0 | 8.10% |
| `arm_elevation` | 55.0 | 8.10% |
| `waist_bends` | 53.0 | 7.81% |
| `jump_front_back` | 19.0 | 2.80% |

> [!NOTE]
> 11 classes are relatively balanced (53 to 67 samples each, ~7.8% - 9.9%).
> **`jump_front_back` is underrepresented** with only 19 samples (2.80%). Stratified splitting is essential.

## 3. Data Hygiene & Missing Value Diagnosis
- **Train Missing / Corrupted Cells**: 638842 / 3904250 (16.36%).
  - Causes identified: Synthetic corruption injected into `train.csv`:
    1. European comma decimal notation (e.g. `'-9,31'`, `'-13,54'`).
    2. Whitespace padding around numeric strings.
    3. Sentinels/tokens: `'?'`, `'-'`, `'unknown'` (coerced to NaN).
    4. Outlier sentinel values: `9999` / `9999.0` (treated as missing values).
  - Uniform distribution: Missing rate is virtually identical across all 23 channels (~16.34% - 16.41%) and all rows (~941 cells per row).
  - Resolution: Cleaned via regex/string formatting and reconstructed using **linear interpolation along the 250-timestep time series axis** with boundary backfill/forward fill (resulting in 0 NaNs).
- **Test Data**: 100% clean float64, 0 NaNs, 0 sentinel values, 0 near-constant columns.

## 4. Per-Channel Statistics & Distribution Shift (Train vs Test)

| Channel | Train Mean | Test Mean | Mean Shift | Train Std | Test Std | Std Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chest_acc_x` | -7.4264 | -7.5893 | 0.1629 | 13.3226 | 5.6760 | 2.35 |
| `chest_acc_y` | -0.1274 | 0.0240 | 0.1514 | 7.0584 | 1.9213 | 3.67 |
| `chest_acc_z` | -1.0564 | -0.5454 | 0.5110 | 10.6498 | 4.9761 | 2.14 |
| `chest_ecg_1` | -0.0019 | -0.0246 | 0.0227 | 2.0724 | 0.6861 | 3.02 |
| `chest_ecg_2` | -0.0205 | -0.0153 | 0.0052 | 1.9495 | 0.7237 | 2.69 |
| `ankle_acc_x` | 2.0119 | 1.5836 | 0.4282 | 9.9123 | 4.2766 | 2.32 |
| `ankle_acc_y` | -9.0967 | -8.9639 | 0.1328 | 12.0724 | 5.2828 | 2.29 |
| `ankle_acc_z` | -0.4053 | -1.2486 | 0.8433 | 14.9634 | 6.4694 | 2.31 |
| `ankle_gyro_x` | 0.1577 | 0.0774 | 0.0804 | 0.9799 | 0.4836 | 2.03 |
| `ankle_gyro_y` | -0.5171 | -0.6115 | 0.0944 | 1.0450 | 0.3897 | 2.68 |
| `ankle_gyro_z` | -0.1350 | -0.1057 | 0.0293 | 1.3759 | 0.5225 | 2.63 |
| `ankle_mag_x` | 0.8785 | 0.0911 | 0.7874 | 181.9001 | 73.5758 | 2.47 |
| `ankle_mag_y` | -2.0713 | -1.9819 | 0.0894 | 106.5894 | 48.8874 | 2.18 |
| `ankle_mag_z` | 0.0376 | -0.5057 | 0.5433 | 38.6822 | 24.4759 | 1.58 |
| `arm_acc_x` | -3.5009 | -3.3322 | 0.1687 | 14.2754 | 5.6334 | 2.53 |
| `arm_acc_y` | -5.9092 | -5.7270 | 0.1822 | 14.8023 | 6.7416 | 2.20 |
| `arm_acc_z` | 2.1816 | 2.9479 | 0.7663 | 10.1609 | 3.9817 | 2.55 |
| `arm_gyro_x` | -0.1677 | -0.2531 | 0.0854 | 1.3071 | 0.4800 | 2.72 |
| `arm_gyro_y` | -0.4547 | -0.3677 | 0.0870 | 1.2829 | 0.5611 | 2.29 |
| `arm_gyro_z` | 0.3498 | 0.4661 | 0.1163 | 1.1760 | 0.5372 | 2.19 |
| `arm_mag_x` | -0.2121 | -0.9375 | 0.7254 | 80.8895 | 32.7930 | 2.47 |
| `arm_mag_y` | 1.6599 | 1.7110 | 0.0510 | 65.9258 | 32.7065 | 2.02 |
| `arm_mag_z` | -1.0606 | 1.0085 | 2.0690 | 181.4521 | 89.7470 | 2.02 |

## 5. Train-vs-Test Adversarial Validation & Shifting Channels
- **Adversarial Validation ROC-AUC**: **1.0000**
- **Key Findings on Distribution Shift**:
  1. **Magnetometer & Extreme Min/Max**: The top features distinguishing train from test are absolute magnetometer extremes and signal boundaries (`ankle_gyro_z_max`, `arm_mag_x_max`, `ankle_mag_x_min`, `arm_acc_y_min`).
  2. **Subject Offset**: Because test data comprises 3 completely different subjects, static baseline sensor offsets shift substantially between train and test.
  3. **Direct Feature Choice Implication**: A robust feature set (Feature Set B) MUST de-emphasize or eliminate raw static channel means/offsets and focus on **dynamic features** (standard deviation, energy, spectral entropy, band power ratios, cross-channel axis correlations, and signal magnitude area SMA).

## 6. Artifact Links
- Class Distribution: `outputs/class_distribution.png`
- Channel Shift Analysis: `outputs/channel_shift.png`
