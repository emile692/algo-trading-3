"""train_meta_xgboost.py"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    f1_score, fbeta_score, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.utils import class_weight
import shap
from tools.logger import setup_logger


# ---------- Hyperparams (config simples) ----------
BETA_FOR_ERRORS = 2.0
MIN_PREC0 = 0.25
MIN_RECALL0_FOR_CONSTRAINT = 0.05


def optimize_threshold_for_error_recall(y_true, prob_class0, beta=BETA_FOR_ERRORS,
                                        min_precision0=MIN_PREC0,
                                        min_recall0=MIN_RECALL0_FOR_CONSTRAINT):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_fbeta = -1.0
    best_threshold_fbeta = 0.5

    y_true_err_as_pos = (y_true == 0).astype(int)
    for thresh in thresholds:
        y_pred_err_flag = (prob_class0 >= thresh).astype(int)
        fbeta = fbeta_score(y_true_err_as_pos, y_pred_err_flag, beta=beta)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold_fbeta = thresh

    prec, rec, thr = precision_recall_curve(y_true_err_as_pos, prob_class0)
    thr_full = np.r_[thr, 1.0]
    ok = (prec >= min_precision0) & (rec >= min_recall0)
    if np.any(ok):
        best_idx = None
        best_fbeta_constrained = -1.0
        for i in np.where(ok)[0]:
            t = thr_full[i]
            y_pred_err_flag = (prob_class0 >= t).astype(int)
            fbeta_i = fbeta_score(y_true_err_as_pos, y_pred_err_flag, beta=beta)
            if fbeta_i > best_fbeta_constrained:
                best_fbeta_constrained = fbeta_i
                best_idx = i
        return float(thr_full[best_idx])
    return float(best_threshold_fbeta)


def train_and_test_meta_xgboost(seed, logger):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    meta_dataset_path = os.path.join(project_root, 'meta_model', 'dataset', f'meta_EURUSD_H1_seed_{seed}.csv')

    df = pd.read_csv(meta_dataset_path)
    class_dist = df["meta_label"].value_counts(normalize=True)
    logger.info(f"Distribution classes méta : 0={class_dist[0]:.2%}, 1={class_dist[1]:.2%}")

    class_weights = class_weight.compute_sample_weight('balanced', y=df['meta_label'])
    sample_weights = pd.Series(class_weights, index=df.index)

    X = df.drop(columns=["y_true", "y_pred", "meta_label", "time"])
    y = df["meta_label"]
    y_true_full = df["y_true"].to_numpy()
    y_pred_full = df["y_pred"].to_numpy()
    time_full = df["time"].to_numpy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    sw_train, sw_test = sample_weights.iloc[:split_idx], sample_weights.iloc[split_idx:]
    y_true_test = y_true_full[split_idx:]
    y_pred_exo_test = y_pred_full[split_idx:]
    time_test = time_full[split_idx:]

    # === Optimisation du seuil ===
    logger.info("Optimisation du seuil pour détection d'erreurs...")
    thresholds = []
    tscv = TimeSeriesSplit(n_splits=3)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        fold_model = XGBClassifier(
            n_estimators=1000, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
            early_stopping_rounds=50, random_state=seed,
        )
        fold_model.fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx],
            sample_weight=sw_train.iloc[train_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            verbose=False
        )
        y_prob_val_class0 = fold_model.predict_proba(X_train.iloc[val_idx])[:, 0]
        optimal_threshold = optimize_threshold_for_error_recall(
            y_train.iloc[val_idx], y_prob_val_class0
        )
        thresholds.append(optimal_threshold)
        logger.info(f"Fold {fold+1}: seuil optimisé = {optimal_threshold:.4f}")

    best_threshold = float(np.mean(thresholds))
    logger.info(f"Seuil optimal moyen: {best_threshold:.4f}")

    # === Entraînement final + calibration ===
    logger.info("Entraînement final du méta-modèle XGBoost...")
    final_model = XGBClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        random_state=seed, eval_metric='aucpr',
    )
    final_model.fit(X_train, y_train, sample_weight=sw_train)
    calibrated_model = CalibratedClassifierCV(final_model, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
    calibrated_model.fit(X_train, y_train, sample_weight=sw_train)

    # === Évaluation ===
    y_prob_class0 = calibrated_model.predict_proba(X_test)[:, 0]
    y_pred_error_flag = (y_prob_class0 >= best_threshold).astype(int)
    y_pred_labels = np.where(y_pred_error_flag == 1, 0, 1)
    roc_auc = roc_auc_score((y_test == 0).astype(int), y_prob_class0)

    logger.info("=== Évaluation méta ===")
    logger.info(classification_report(y_test, y_pred_labels))
    logger.info(f"ROC AUC = {roc_auc:.4f}")
    logger.info(confusion_matrix(y_test, y_pred_labels))

    # === Feature importance (SHAP) ===
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_sum = shap_df.abs().mean().sort_values(ascending=False)
    logger.info("Top features SHAP:")
    for feat, imp in shap_sum.head(15).items():
        logger.info(f"  {feat}: {imp:.4f}")

    # === Sauvegarde ===
    results_dir = os.path.join(project_root, 'meta_model', 'results', f'seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)

    joblib.dump({'model': calibrated_model, 'threshold': float(best_threshold)},
                os.path.join(results_dir, f"xgboost_meta_model_seed_{seed}.joblib"))
    np.save(os.path.join(results_dir, 'y_prob_class0.npy'), y_prob_class0)
    np.save(os.path.join(results_dir, 'y_true.npy'), y_true_test)
    np.save(os.path.join(results_dir, 'y_pred_exo.npy'), y_pred_exo_test)
    np.save(os.path.join(results_dir, 'y_pred_meta.npy'), y_pred_labels)
    np.save(os.path.join(results_dir, 'time.npy'), time_test)

    with open(os.path.join(results_dir, 'meta_metrics.json'), 'w') as f:
        json.dump({'roc_auc': float(roc_auc), 'threshold': float(best_threshold)}, f, indent=4)

    logger.info(f"✅ Modèle sauvegardé dans {results_dir}")
    return roc_auc


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("=== Entraînement et test du méta-modèle XGBoost ===")
    train_and_test_meta_xgboost(42, logger)
