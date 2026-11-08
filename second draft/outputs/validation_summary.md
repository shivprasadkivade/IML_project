# Validation Strategy Summary: Pseudo-Subject Group CV

## 1. Challenge & Context
- **True Subject IDs are omitted** in both train and test sets.
- The training set comprises windows from **5 subjects**, whereas the test set comprises **3 completely different subjects**.
- Random K-Fold CV splits samples from the same subject across train and validation folds, causing **severe data leakage** and overly optimistic CV scores.

## 2. Row Order & ID Investigation
- Sequential runs in original file order: 626 runs across 679 samples.
- Sequential runs when sorted by ID: 581 runs across 679 samples.
- Findings: Data has been permuted. No monolithic contiguous subject blocks exist in ID or file order.

## 3. Pseudo-Subject Grouping Methodology
- Extracted **static anatomical and mounting baseline features**: per-window means of 3-axis magnetometers (ankle, arm) and 2-lead ECG baselines.
- Applied `StandardScaler` followed by `KMeans(n_clusters=5, random_state=42, n_init=30)`.
- **Group Sizes**:
  - Group 0: 206 windows (30.3%)
  - Group 1: 149 windows (21.9%)
  - Group 2: 116 windows (17.1%)
  - Group 3: 21 windows (3.1%)
  - Group 4: 187 windows (27.5%)
- **Class Diversity**: All 12 classes are present across multiple pseudo-subject groups (minimum representation: 4 groups per class).

## 4. Primary CV Protocol
- **Primary Cross-Validator**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` using the 5 pseudo-subject groups.
- **Model Selection Policy**: All hyperparameter tuning, feature set comparisons, and ensemble weights MUST be chosen solely based on the pseudo-subject Group CV score.
- **Leakage Gap Metric**: Standard `StratifiedKFold(n_splits=5)` is tracked concurrently to measure and report the exact optimistic leakage gap.
