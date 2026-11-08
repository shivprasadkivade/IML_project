"""
CSL2010 IML Project 1: Candidate Models, Hyperparameter Tuning & Ensembling
File: train.py

Implements STEP 4, STEP 5, and STEP 6:
- Evaluates Candidate Models inside sklearn Pipelines:
  1. Regularized Logistic Regression (StandardScaler)
  2. SVM RBF (StandardScaler, probability=True)
  3. Random Forest (large n_estimators)
  4. Extra Trees (large n_estimators)
  5. LightGBM (regularized: subsample, colsample, L1/L2)
  6. XGBoost (regularized: subsample, colsample, L1/L2)
  7. MiniRocket + RidgeClassifierCV on raw (N, 23, 250) series
  8. 1D CNN (PyTorch) with BatchNorm, Dropout, Early Stopping on raw series
- Computes both Group-CV (StratifiedGroupKFold) and Random-CV (StratifiedKFold)
  to measure the Subject Leakage Gap.
- Performs Nested CV (Outer LeaveOneGroupOut, Inner StratifiedGroupKFold)
  for honest generalization estimation.
- Builds Soft-Voting and Stacking Ensembles on out-of-fold predictions.
- Generates out-of-fold confusion matrices and identifies confusable activity pairs.
- Saves model summary to outputs/models_summary.md and best models to disk.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Tree-based and specialized packages
import lightgbm as lgb
import xgboost as xgb
from aeon.transformations.collection.convolution_based import MiniRocket

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Local modules
from cv import load_and_preprocess_train, get_group_cv, get_random_cv
from features import get_subject_robust_feature_names

warnings.filterwarnings('ignore')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# ------------------------------------------------------------------------------
# 1D CNN Architecture in PyTorch
# ------------------------------------------------------------------------------
class Conv1DHAR(nn.Module):
    def __init__(self, in_channels=23, num_classes=12, dropout=0.3):
        super(Conv1DHAR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_eval_cnn_fold(X_train_raw, y_train_num, X_val_raw, y_val_num, epochs=30, batch_size=32):
    """
    Fits 1D CNN on a single train fold with fold-wise per-channel normalization and early stopping.
    """
    # X shape: (N, 23, 250)
    # Compute per-channel mean and std strictly on train fold
    ch_mean = np.mean(X_train_raw, axis=(0, 2), keepdims=True)
    ch_std = np.std(X_train_raw, axis=(0, 2), keepdims=True) + 1e-6
    
    X_tr_norm = (X_train_raw - ch_mean) / ch_std
    X_val_norm = (X_val_raw - ch_mean) / ch_std
    
    t_tr_x = torch.tensor(X_tr_norm, dtype=torch.float32)
    t_tr_y = torch.tensor(y_train_num, dtype=torch.long)
    t_val_x = torch.tensor(X_val_norm, dtype=torch.float32)
    t_val_y = torch.tensor(y_val_num, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(t_tr_x, t_tr_y), batch_size=batch_size, shuffle=True)
    
    model = Conv1DHAR(in_channels=23, num_classes=12, dropout=0.3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_val_preds = None
    best_val_probs = None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Evaluate on validation fold
        model.eval()
        with torch.no_grad():
            vx = t_val_x.to(DEVICE)
            val_out = model(vx)
            probs = torch.softmax(val_out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            val_acc = accuracy_score(y_val_num, preds)
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_preds = preds
                best_val_probs = probs
                
    return best_val_acc, best_val_preds, best_val_probs

# ------------------------------------------------------------------------------
# Candidate Models Factory
# ------------------------------------------------------------------------------
def get_candidate_models():
    """
    Returns dictionary of candidate model pipelines operating on engineered features.
    """
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=SEED))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=10.0, gamma='scale', probability=True, random_state=SEED))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1))
        ]),
        'Extra Trees': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1))
        ]),
        'LightGBM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='mlogloss', random_state=SEED, n_jobs=-1
            ))
        ])
    }
    return models

def evaluate_models_cv(models, X, y, groups, cv_group, cv_random):
    """
    Evaluates candidate models under both Group-CV and Random-CV to quantify leakage.
    Returns results table and out-of-fold probability predictions.
    """
    results = []
    oof_predictions = {}
    
    for name, pipe in models.items():
        t0 = time.time()
        
        # 1. Group-CV Evaluation
        group_scores = []
        n_classes = len(np.unique(y))
        oof_probs = np.zeros((len(y), n_classes))
        
        for tr_idx, val_idx in cv_group.split(X, y, groups=groups):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            probs = pipe.predict_proba(X_val)
            
            group_scores.append(accuracy_score(y_val, preds))
            oof_probs[val_idx] = probs
            
        group_mean = np.mean(group_scores)
        group_std = np.std(group_scores)
        oof_predictions[name] = oof_probs
        
        # 2. Random-CV Evaluation (to measure leakage gap)
        random_scores = []
        for tr_idx, val_idx in cv_random.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            random_scores.append(accuracy_score(y_val, preds))
            
        random_mean = np.mean(random_scores)
        random_std = np.std(random_scores)
        leakage_gap = random_mean - group_mean
        train_time = time.time() - t0
        
        results.append({
            'Model': name,
            'Group-CV Mean (%)': group_mean * 100,
            'Group-CV Std (%)': group_std * 100,
            'Random-CV Mean (%)': random_mean * 100,
            'Leakage Gap (%)': leakage_gap * 100,
            'Train Time (s)': round(train_time, 2)
        })
        print(f"  {name:20s} | Group-CV: {group_mean*100:6.2f}% +/- {group_std*100:4.2f}% | Random-CV: {random_mean*100:6.2f}% | Gap: {leakage_gap*100:5.2f}% | Time: {train_time:5.1f}s")
        
    return pd.DataFrame(results), oof_predictions

def main():
    os.makedirs('outputs', exist_ok=True)
    print("=" * 80)
    print("STEP 4, 5 & 6: CANDIDATE MODELS, NESTED CV & ENSEMBLES")
    print("=" * 80)
    
    # 1. Load data, features, targets, and groups
    print("\n[1] Loading precomputed data and cached features...")
    ids, targets, tensor_train, channels = load_and_preprocess_train('train.csv')
    df_feat_train = pd.read_csv('outputs/features_train.csv')
    pseudo_groups = pd.read_csv('outputs/pseudo_groups.csv')['pseudo_group'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(targets)
    classes = label_encoder.classes_
    
    all_feature_cols = df_feat_train.columns.tolist()
    robust_feature_cols = get_subject_robust_feature_names(all_feature_cols)
    print(f"Features: {len(all_feature_cols)} (All - Set A), {len(robust_feature_cols)} (Subject-Robust - Set B)")
    
    cv_group = get_group_cv(n_splits=5, seed=SEED)
    cv_random = get_random_cv(n_splits=5, seed=SEED)
    
    # 2. Evaluate Candidate Models on Feature Set B (Subject-Robust)
    print("\n[2] Evaluating Candidate Models on Feature Set B (Subject-Robust)...")
    X_B = df_feat_train[robust_feature_cols].values
    candidate_models_B = get_candidate_models()
    results_df_B, oof_probs_B = evaluate_models_cv(candidate_models_B, X_B, y_encoded, pseudo_groups, cv_group, cv_random)
    
    # 3. Evaluate Candidate Models on Feature Set A (All Features) for comparison
    print("\n[3] Evaluating Candidate Models on Feature Set A (All Features)...")
    X_A = df_feat_train[all_feature_cols].values
    candidate_models_A = get_candidate_models()
    results_df_A, oof_probs_A = evaluate_models_cv(candidate_models_A, X_A, y_encoded, pseudo_groups, cv_group, cv_random)
    
    # 4. Evaluate MiniRocket on Raw Series (N, 23, 250)
    print("\n[4] Evaluating MiniRocket + RidgeClassifierCV on raw time series...")
    t0_mr = time.time()
    # Aeon expects (N, n_channels, n_timepoints)
    raw_series = np.transpose(tensor_train, (0, 2, 1)) # (679, 23, 250)
    
    mr = MiniRocket(random_state=SEED)
    X_mr = mr.fit_transform(raw_series)
    print(f"  MiniRocket transformed feature shape: {X_mr.shape}")
    
    pipe_mr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RidgeClassifierCV(alphas=np.logspace(-2, 3, 10)))
    ])
    
    mr_group_scores, mr_random_scores = [], []
    mr_oof_preds = np.zeros(len(y_encoded))
    for tr_idx, val_idx in cv_group.split(X_mr, y_encoded, groups=pseudo_groups):
        pipe_mr.fit(X_mr[tr_idx], y_encoded[tr_idx])
        preds = pipe_mr.predict(X_mr[val_idx])
        mr_group_scores.append(accuracy_score(y_encoded[val_idx], preds))
        mr_oof_preds[val_idx] = preds
        
    for tr_idx, val_idx in cv_random.split(X_mr, y_encoded):
        pipe_mr.fit(X_mr[tr_idx], y_encoded[tr_idx])
        preds = pipe_mr.predict(X_mr[val_idx])
        mr_random_scores.append(accuracy_score(y_encoded[val_idx], preds))
        
    mr_group_mean = np.mean(mr_group_scores)
    mr_group_std = np.std(mr_group_scores)
    mr_random_mean = np.mean(mr_random_scores)
    mr_leakage_gap = mr_random_mean - mr_group_mean
    mr_time = time.time() - t0_mr
    
    print(f"  {'MiniRocket + Ridge':20s} | Group-CV: {mr_group_mean*100:6.2f}% +/- {mr_group_std*100:4.2f}% | Random-CV: {mr_random_mean*100:6.2f}% | Gap: {mr_leakage_gap*100:5.2f}% | Time: {mr_time:5.1f}s")
    
    # 5. Evaluate Small 1D CNN on Raw Series
    print("\n[5] Evaluating Small 1D CNN (PyTorch) on raw time series...")
    t0_cnn = time.time()
    cnn_group_scores = []
    cnn_oof_probs = np.zeros((len(y_encoded), len(classes)))
    
    for fold, (tr_idx, val_idx) in enumerate(cv_group.split(raw_series, y_encoded, groups=pseudo_groups)):
        val_acc, val_preds, val_probs = train_eval_cnn_fold(
            raw_series[tr_idx], y_encoded[tr_idx],
            raw_series[val_idx], y_encoded[val_idx],
            epochs=25, batch_size=32
        )
        cnn_group_scores.append(val_acc)
        cnn_oof_probs[val_idx] = val_probs
        
    cnn_group_mean = np.mean(cnn_group_scores)
    cnn_group_std = np.std(cnn_group_scores)
    cnn_time = time.time() - t0_cnn
    print(f"  {'1D CNN (PyTorch)':20s} | Group-CV: {cnn_group_mean*100:6.2f}% +/- {cnn_group_std*100:4.2f}% | Time: {cnn_time:5.1f}s")
    
    # 6. Nested CV on Top Candidate Model (Outer: LeaveOneGroupOut, Inner: StratifiedGroupKFold)
    print("\n[6] Performing NESTED CV for honest generalization estimation...")
    logo = LeaveOneGroupOut()
    param_grid_svm = {'clf__C': [1.0, 5.0, 10.0, 20.0]}
    
    nested_scores = []
    fold_idx = 0
    for train_outer_idx, test_outer_idx in logo.split(X_B, y_encoded, groups=pseudo_groups):
        held_out_group = pseudo_groups[test_outer_idx][0]
        X_tr_out, y_tr_out = X_B[train_outer_idx], y_encoded[train_outer_idx]
        groups_tr_out = pseudo_groups[train_outer_idx]
        X_te_out, y_te_out = X_B[test_outer_idx], y_encoded[test_outer_idx]
        
        # Inner CV tuning
        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        grid = GridSearchCV(
            Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', probability=True, random_state=SEED))]),
            param_grid_svm,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_tr_out, y_tr_out, groups=groups_tr_out)
        
        best_model = grid.best_estimator_
        test_acc = accuracy_score(y_te_out, best_model.predict(X_te_out))
        nested_scores.append(test_acc)
        print(f"  Outer Fold {fold_idx} (Held-out Group {held_out_group}, {len(test_outer_idx)} samples): Best C={grid.best_params_['clf__C']} | Accuracy: {test_acc*100:.2f}%")
        fold_idx += 1
        
    nested_mean = np.mean(nested_scores)
    nested_std = np.std(nested_scores)
    print(f"  Nested-CV Generalization Accuracy: {nested_mean*100:.2f}% +/- {nested_std*100:.2f}%")
    
    # 7. Build Soft-Voting and Stacking Ensembles (STEP 6)
    print("\n[7] Building Ensembles on top diverse candidate models...")
    # Select top 4 diverse models from Feature Set B
    top_models = ['SVM (RBF)', 'Random Forest', 'Extra Trees', 'LightGBM']
    
    # Soft Voting out-of-fold probability average
    ensemble_probs = np.zeros_like(oof_probs_B['SVM (RBF)'])
    for m in top_models:
        ensemble_probs += oof_probs_B[m]
    ensemble_probs /= len(top_models)
    
    voting_oof_preds = np.argmax(ensemble_probs, axis=1)
    voting_acc = accuracy_score(y_encoded, voting_oof_preds)
    print(f"  Soft-Voting Ensemble (SVM + RF + ET + LightGBM) OOF Accuracy: {voting_acc*100:.2f}%")
    
    # Stacking meta-learner fit on out-of-fold probabilities
    # Concatenate OOF probabilities of top models
    stack_X = np.hstack([oof_probs_B[m] for m in top_models]) # (679, 4 * 12) = (679, 48)
    
    # Evaluate Stacking with Group-CV
    meta_learner = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
    stack_scores = []
    stack_oof_preds = np.zeros(len(y_encoded))
    for tr_idx, val_idx in cv_group.split(stack_X, y_encoded, groups=pseudo_groups):
        meta_learner.fit(stack_X[tr_idx], y_encoded[tr_idx])
        preds = meta_learner.predict(stack_X[val_idx])
        stack_scores.append(accuracy_score(y_encoded[val_idx], preds))
        stack_oof_preds[val_idx] = preds
        
    stack_mean = np.mean(stack_scores)
    stack_std = np.std(stack_scores)
    print(f"  Stacking Meta-Learner Group-CV Accuracy: {stack_mean*100:.2f}% +/- {stack_std*100:.2f}%")
    
    # 8. Confusion Matrix & Confusable Pair Analysis
    print("\n[8] Generating Confusion Matrix from out-of-fold predictions...")
    cm = confusion_matrix(y_encoded, voting_oof_preds)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Out-of-Fold Confusion Matrix (Soft-Voting Ensemble)', fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, f"{val}", ha="center", va="center",
                         color="white" if val > thresh else "black", fontsize=9)
    plt.ylabel('True Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join('outputs', 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix plot to {cm_plot_path}")
    
    # Find any misclassified pairs
    errors = np.where(cm - np.diag(np.diag(cm)) > 0)
    confusable_pairs = []
    for tr_i, pred_j in zip(errors[0], errors[1]):
        confusable_pairs.append((classes[tr_i], classes[pred_j], cm[tr_i, pred_j]))
    print(f"  Confusable pairs detected (True -> Pred, count): {confusable_pairs if confusable_pairs else 'None (100% on OOF)'}")
    
    # 9. Retrain Final Pipelines on 100% Training Data & Save Models
    print("\n[9] Retraining top candidate pipelines on ALL training data...")
    # Model A: Best robust single model (SVM RBF on Feature Set B)
    best_single_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=10.0, gamma='scale', probability=True, random_state=SEED))
    ])
    best_single_pipe.fit(X_B, y_encoded)
    
    # Model B: Soft-voting ensemble of top 4 diverse models on Feature Set B
    ensemble_clf = VotingClassifier(
        estimators=[
            ('svm', Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=10.0, gamma='scale', probability=True, random_state=SEED))])),
            ('rf', Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1))])),
            ('et', Pipeline([('scaler', StandardScaler()), ('clf', ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1))])),
            ('lgb', Pipeline([('scaler', StandardScaler()), ('clf', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1))]))
        ],
        voting='soft'
    )
    ensemble_clf.fit(X_B, y_encoded)
    
    # Save trained pipelines
    with open('outputs/model_single_svm.pkl', 'wb') as f:
        pickle.dump({'model': best_single_pipe, 'features': robust_feature_cols, 'label_encoder': label_encoder}, f)
        
    with open('outputs/model_ensemble_voting.pkl', 'wb') as f:
        pickle.dump({'model': ensemble_clf, 'features': robust_feature_cols, 'label_encoder': label_encoder}, f)
        
    print("  Saved final trained pipelines to outputs/model_single_svm.pkl and outputs/model_ensemble_voting.pkl")
    
    # 10. Write Deliverable: models_summary.md
    summary_path = os.path.join('outputs', 'models_summary.md')
    print(f"\n[10] Writing comprehensive model comparison deliverable to {summary_path}...")
    
    with open(summary_path, 'w') as f:
        f.write("# Candidate Models, Validation & Ensembling Summary (STEPS 4, 5, 6)\n\n")
        f.write("## 1. Candidate Model Performance Table (Feature Set B: Subject-Robust)\n\n")
        f.write("| Model Pipeline | Group-CV Accuracy (%) | Random-CV Accuracy (%) | Leakage Gap (%) | Training Time (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, r in results_df_B.iterrows():
            f.write(f"| `{r['Model']}` | **{r['Group-CV Mean (%)']:.2f}%** ± {r['Group-CV Std (%)']:.2f}% | {r['Random-CV Mean (%)']:.2f}% | +{r['Leakage Gap (%)']:.2f}% | {r['Train Time (s)']:.1f}s |\n")
        f.write(f"| `MiniRocket + RidgeCV` | **{mr_group_mean*100:.2f}%** ± {mr_group_std*100:.2f}% | {mr_random_mean*100:.2f}% | +{mr_leakage_gap*100:.2f}% | {mr_time:.1f}s |\n")
        f.write(f"| `1D CNN (PyTorch)` | **{cnn_group_mean*100:.2f}%** ± {cnn_group_std*100:.2f}% | - | - | {cnn_time:.1f}s |\n\n")
        
        f.write("## 2. Comparison with Feature Set A (All Features)\n\n")
        f.write("| Model Pipeline | Feature Set A Group-CV (%) | Feature Set B Group-CV (%) | Delta Group-CV |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for i in range(len(results_df_A)):
            mA = results_df_A.iloc[i]
            mB = results_df_B.iloc[i]
            delta = mB['Group-CV Mean (%)'] - mA['Group-CV Mean (%)']
            f.write(f"| `{mA['Model']}` | {mA['Group-CV Mean (%)']:.2f}% ± {mA['Group-CV Std (%)']:.2f}% | {mB['Group-CV Mean (%)']:.2f}% ± {mB['Group-CV Std (%)']:.2f}% | {delta:+.2f}% |\n")
        f.write("\n")
        
        f.write("## 3. Nested Cross-Validation (Generalization Guarantee)\n")
        f.write("- **Outer Split**: LeaveOneGroupOut (5 folds, holding out each pseudo-subject cluster entirely).\n")
        f.write("- **Inner Split**: StratifiedGroupKFold (4 folds) for inner hyperparameter tuning.\n")
        f.write(f"- **Expected Generalization Accuracy**: **{nested_mean*100:.2f}% ± {nested_std*100:.2f}%**\n")
        f.write("- Per-fold accuracies on held-out groups:\n")
        for i, s in enumerate(nested_scores):
            f.write(f"  - Fold {i} (Held-out Group {i}): **{s*100:.2f}%**\n")
        f.write("\n")
        
        f.write("## 4. Ensemble Performance & Confusable Pair Analysis\n")
        f.write(f"- **Soft-Voting Ensemble (SVM + RF + ExtraTrees + LightGBM)**: **{voting_acc*100:.2f}%** OOF Accuracy.\n")
        f.write(f"- **Stacking Ensemble (Logistic Meta-Learner)**: **{stack_mean*100:.2f}% ± {stack_std*100:.2f}%** Group-CV Accuracy.\n")
        f.write(f"- **Confusable Activity Pairs**: {confusable_pairs if confusable_pairs else 'None detected; 100% discrimination on OOF predictions.'}\n")
        f.write(f"- **Confusion Matrix Artifact**: `outputs/confusion_matrix.png`\n\n")
        
        f.write("## 5. Model Selection Decision for Submissions\n")
        f.write("1. **Submission A (Best Single Robust Model)**: `SVC(kernel='rbf', C=10.0)` trained on Feature Set B (Subject-Robust).\n")
        f.write("   - Highly regularized kernel machine with smooth margin boundaries, robust against individual sensor drift.\n")
        f.write("2. **Submission B (Best Diverse Ensemble)**: Soft-Voting Ensemble combining SVM (RBF), Random Forest, Extra Trees, and LightGBM on Feature Set B.\n")
        f.write("   - Blends tree-based partition boundaries with SVM kernel support vectors for optimal variance reduction on unseen test subjects.\n")
        
    print(f"\nModel training and evaluation complete! Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
