"""
CSL2010 IML Project 1: Final Prediction & Submission Generation
File: predict.py

Implements STEP 7:
1. Loads cached test features and trained models:
   - Candidate Submission A: Best Single Robust Model (Extra Trees on Feature Set B)
   - Candidate Submission B: Best Diverse Ensemble (Soft-Voting Ensemble: SVM + RF + ET + LightGBM on Feature Set B)
2. Predicts test.csv window activity classes.
3. Formats predictions into submission_A.csv and submission_B.csv.
4. Executes strict validation asserts (row count, exact ID alignment, valid label vocabulary, no NaNs).
5. Compares test prediction distributions against train class distributions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import lightgbm as lgb

SEED = 42

def validate_submission_format(sub_df, sample_sub_df, sub_name="Submission"):
    """
    Validates submission against competition requirements:
    1. Exactly 2 columns: 'id' and 'target'
    2. Exact row count match (403 rows)
    3. Exact ID match and ordering
    4. No missing/null values
    5. All labels belong to the 12 known competition classes
    """
    print(f"\n[Validation] Checking {sub_name}...")
    errors = []
    
    # 1. Column names
    expected_cols = ['id', 'target']
    if list(sub_df.columns) != expected_cols:
        errors.append(f"Columns mismatch! Expected {expected_cols}, got {list(sub_df.columns)}")
        
    # 2. Row count
    if len(sub_df) != len(sample_sub_df):
        errors.append(f"Row count mismatch! Expected {len(sample_sub_df)}, got {len(sub_df)}")
        
    # 3. Exact ID alignment
    if not (sub_df['id'] == sample_sub_df['id']).all():
        errors.append("IDs do not match sample_submission.csv exactly or row ordering is altered!")
        
    # 4. Nulls
    nan_count = sub_df.isna().sum().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN/null values in submission!")
        
    # 5. Label validity
    valid_classes = {
        'arm_elevation', 'climbing_stairs', 'cycling', 'jogging',
        'jump_front_back', 'knees_bending', 'lying', 'running',
        'sitting', 'standing', 'waist_bends', 'walking'
    }
    invalid_labels = set(sub_df['target'].unique()) - valid_classes
    if len(invalid_labels) > 0:
        errors.append(f"Detected invalid labels not in train target vocabulary: {invalid_labels}")
        
    if len(errors) == 0:
        print(f"  --> {sub_name}: VALIDATION PASS! (403 rows, valid IDs, 0 nulls, correct classes)")
        return True
    else:
        print(f"  --> {sub_name}: VALIDATION FAIL!")
        for e in errors:
            print(f"      - {e}")
        return False

def main():
    os.makedirs('outputs', exist_ok=True)
    print("=" * 80)
    print("STEP 7: FINAL MODEL FITTING, PREDICTION & SUBMISSION GENERATION")
    print("=" * 80)
    
    # 1. Load data & features
    print("\n[1] Loading training targets, sample submission, and precomputed features...")
    train_df = pd.read_csv('train.csv', low_memory=False)
    sample_sub = pd.read_csv('sample_submission.csv')
    df_feat_train = pd.read_csv('outputs/features_train.csv')
    df_feat_test = pd.read_csv('outputs/features_test.csv')
    
    from features import get_subject_robust_feature_names
    robust_features = get_subject_robust_feature_names(df_feat_train.columns.tolist())
    print(f"Training windows: {len(train_df)}, Test windows: {len(sample_sub)}")
    print(f"Subject-Robust Feature count: {len(robust_features)}")
    
    X_train = df_feat_train[robust_features].values
    y_train = train_df['target'].values
    X_test = df_feat_test[robust_features].values
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    
    # 2. Fit Model A: Best Single Robust Model (Extra Trees on Feature Set B)
    # Extra Trees had the highest Group-CV score (99.66%) and fast, stable predictions
    print("\n[2] Fitting Candidate Model A: Best Single Robust Model (Extra Trees)...")
    pipe_single = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier(n_estimators=500, max_depth=14, random_state=SEED, n_jobs=-1))
    ])
    pipe_single.fit(X_train, y_train_enc)
    
    preds_single_enc = pipe_single.predict(X_test)
    preds_single = label_encoder.inverse_transform(preds_single_enc)
    
    sub_A = pd.DataFrame({'id': sample_sub['id'], 'target': preds_single})
    sub_A_path = 'outputs/submission_A.csv'
    sub_A.to_csv(sub_A_path, index=False)
    print(f"  Saved candidate Submission A to {sub_A_path}")
    
    # 3. Fit Model B: Diverse Soft-Voting Ensemble (SVM + RF + ExtraTrees + LightGBM on Feature Set B)
    print("\n[3] Fitting Candidate Model B: Diverse Soft-Voting Ensemble...")
    pipe_ensemble = VotingClassifier(
        estimators=[
            ('svm', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(C=5.0, gamma='scale', probability=True, random_state=SEED))
            ])),
            ('rf', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=400, max_depth=12, random_state=SEED, n_jobs=-1))
            ])),
            ('et', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', ExtraTreesClassifier(n_estimators=400, max_depth=14, random_state=SEED, n_jobs=-1))
            ])),
            ('lgb', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.04, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=SEED, verbose=-1, n_jobs=-1
                ))
            ]))
        ],
        voting='soft'
    )
    pipe_ensemble.fit(X_train, y_train_enc)
    
    preds_ensemble_enc = pipe_ensemble.predict(X_test)
    preds_ensemble = label_encoder.inverse_transform(preds_ensemble_enc)
    
    sub_B = pd.DataFrame({'id': sample_sub['id'], 'target': preds_ensemble})
    sub_B_path = 'outputs/submission_B.csv'
    sub_B.to_csv(sub_B_path, index=False)
    print(f"  Saved candidate Submission B to {sub_B_path}")
    
    # Also save root copies for user convenience
    sub_A.to_csv('submission_A.csv', index=False)
    sub_B.to_csv('submission_B.csv', index=False)
    print("  Also copied submission_A.csv and submission_B.csv to project root.")
    
    # 4. Strict Validation
    val_A = validate_submission_format(sub_A, sample_sub, "Submission A (Extra Trees)")
    val_B = validate_submission_format(sub_B, sample_sub, "Submission B (Soft-Voting Ensemble)")
    assert val_A and val_B, "Submission validation failed!"
    
    # 5. Prediction Distribution Comparison
    print("\n[5] Comparing Target Distribution: Train vs Test Predictions:")
    train_dist = (pd.Series(y_train).value_counts(normalize=True) * 100).round(2)
    sub_A_dist = (sub_A['target'].value_counts(normalize=True) * 100).round(2)
    sub_B_dist = (sub_B['target'].value_counts(normalize=True) * 100).round(2)
    
    dist_comp = pd.DataFrame({
        'Train (%)': train_dist,
        'Sub A Count': sub_A['target'].value_counts(),
        'Sub A (%)': sub_A_dist,
        'Sub B Count': sub_B['target'].value_counts(),
        'Sub B (%)': sub_B_dist
    }).fillna(0)
    print(dist_comp.to_string())
    
    # Agreement between Submission A and B
    agreement = (sub_A['target'] == sub_B['target']).mean() * 100
    print(f"\nPrediction Agreement between Submission A and Submission B: {agreement:.2f}% ({(sub_A['target'] == sub_B['target']).sum()} / {len(sub_A)} rows match)")
    
    print("\nFinal prediction generation completed successfully!")

if __name__ == '__main__':
    main()
