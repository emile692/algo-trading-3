"""train_test_meta_xgboost.py"""
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


# ---------- Hyperparams (edits MINIMAUX et configurables) ----------
BETA_FOR_ERRORS = 2.0           # F_beta pour favoriser le rappel des erreurs
MIN_PREC0 = 0.25             # contrainte : précision minimale demandée sur la classe 0 (erreurs)
MIN_RECALL0_FOR_CONSTRAINT = 0.05  # éviter un seuil trivial qui coupe tout

def optimize_threshold_for_error_recall(y_true, prob_class0, beta=BETA_FOR_ERRORS,
                                        min_precision0=MIN_PREC0,
                                        min_recall0=MIN_RECALL0_FOR_CONSTRAINT):
    """
    Optimise le seuil sur prob_class0 (proba d'être classe 0 = erreur) en 2 temps :
      1) maximise F_beta (pos_label=1 <=> "prédire erreur")
      2) applique une contrainte de précision mini sur la classe 0 via la PR-curve;
         si aucun seuil ne satisfait la contrainte (avec un rappel minimal), on garde le seuil F_beta.

    y_true: labels {0,1} (0=erreur, 1=correct)
    prob_class0: proba d'être classe 0 (erreur)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_fbeta = -1.0
    best_threshold_fbeta = 0.5

    # Etape 1 : F-beta
    y_true_err_as_pos = (y_true == 0).astype(int)
    for thresh in thresholds:
        y_pred_err_flag = (prob_class0 >= thresh).astype(int)  # 1 = prédire erreur
        fbeta = fbeta_score(y_true_err_as_pos, y_pred_err_flag, beta=beta)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold_fbeta = thresh

    # Etape 2 : contrainte de précision mini pour la classe 0
    # On utilise la PR-curve en considérant "erreur" comme la classe positive
    prec, rec, thr = precision_recall_curve(y_true_err_as_pos, prob_class0)
    # thr est de taille len(prec)-1 ; on aligne
    thr_full = np.r_[thr, 1.0]

    # Sélection des seuils qui respectent la contrainte de précision et un rappel minimum
    ok = (prec >= min_precision0) & (rec >= min_recall0)
    if np.any(ok):
        # Parmi ces seuils admissibles, on choisit celui qui maximise F_beta aussi
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

    # Sinon, on garde le seuil F_beta
    return float(best_threshold_fbeta)


def train_and_test_meta_xgboost(seed, logger):
    # === Chargement du dataset méta ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    meta_dataset_path = os.path.join(project_root, 'meta_model', 'dataset', 'features_and_target',
                                     f"meta_dataset_seed_{seed}.csv")
    df = pd.read_csv(meta_dataset_path)

    # Analyse du déséquilibre de classes
    class_dist = df["meta_label"].value_counts(normalize=True)
    logger.info(
        f"Distribution des classes méta: Classe 0 (erreur): {class_dist[0]:.2%}, Classe 1 (correct): {class_dist[1]:.2%}"
    )

    # Poids de classe (on les conserve via sample_weight)
    class_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=df['meta_label']
    )
    sample_weights = pd.Series(class_weights, index=df.index)

    time_full = df["time"].to_numpy()

    # === Séparation X / y ===
    drop_cols = ["y_true", "y_pred", "meta_label", "time"]
    X = df.drop(columns=drop_cols)
    y = df["meta_label"]
    y_true_full = df["y_true"].to_numpy()
    y_pred_full = df["y_pred"].to_numpy()

    # Clean minimal sur features (stabilité)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === Split temporel 80/20 (pas de fuite) ===
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    sw_train, sw_test = sample_weights.iloc[:split_idx], sample_weights.iloc[split_idx:]
    y_true_train, y_true_test = y_true_full[:split_idx], y_true_full[split_idx:]
    y_pred_train, y_pred_exo_test = y_pred_full[:split_idx], y_pred_full[split_idx:]
    time_train, time_test = time_full[:split_idx], time_full[split_idx:]

    # === Validation croisée pour l'optimisation du seuil ===
    logger.info("Optimisation du seuil pour détection d'erreurs...")
    thresholds = []

    tscv = TimeSeriesSplit(n_splits=3)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        fold_model = XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=seed,
        )

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sw_fold_train = sw_train.iloc[train_idx]

        fold_model.fit(
            X_fold_train, y_fold_train,
            sample_weight=sw_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        # proba classe 0 (erreur)
        y_prob_val_class0 = fold_model.predict_proba(X_fold_val)[:, 0]

        optimal_threshold = optimize_threshold_for_error_recall(
            y_fold_val, y_prob_val_class0,
            beta=BETA_FOR_ERRORS, min_precision0=MIN_PREC0, min_recall0=MIN_RECALL0_FOR_CONSTRAINT
        )

        thresholds.append(optimal_threshold)
        logger.info(f"Fold {fold + 1}: seuil optimisé pour erreurs = {optimal_threshold:.4f}")

    best_threshold = float(np.mean(thresholds))
    logger.info(f"Seuil optimal moyen: {best_threshold:.4f}")

    # Log (informatif) — on ne l'utilise pas directement car pos_label de XGBoost = 1
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    logger.info(f"Scale pos weight (informatif): {scale_pos_weight:.2f} (ratio classe0/classe1)")

    # === Entraînement final avec calibration ===
    logger.info("Entraînement final du méta-modèle...")
    final_model = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=seed,
        eval_metric='aucpr',
    )

    final_model.fit(X_train, y_train, sample_weight=sw_train, eval_set=[(X_train, y_train)], verbose=False)

    logger.info("Calibration du modèle...")
    calibrated_model = CalibratedClassifierCV(final_model, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
    calibrated_model.fit(X_train, y_train, sample_weight=sw_train)

    # Probabilité d'être une erreur (classe 0)
    y_prob_class0 = calibrated_model.predict_proba(X_test)[:, 0]

    # Règle de décision : prédire "erreur" si prob_class0 >= best_threshold
    y_pred_error_flag = (y_prob_class0 >= best_threshold).astype(int)
    # reconvertir vers labels {0,1} : 0 = erreur, 1 = correct
    y_pred_labels = np.where(y_pred_error_flag == 1, 0, 1)

    logger.info("Évaluation détaillée des erreurs:")
    logger.info(classification_report(y_test, y_pred_labels))

    class0_report = classification_report(
        y_test, y_pred_labels,
        target_names=["Erreur (0)", "Correct (1)"],
        output_dict=True
    )["Erreur (0)"]

    class1_report = classification_report(
        y_test, y_pred_labels,
        target_names=["Erreur (0)", "Correct (1)"],
        output_dict=True
    )
    class1_metrics = class1_report["Correct (1)"]

    roc_auc = roc_auc_score((y_test == 0).astype(int), y_prob_class0)
    logger.info(f"ROC AUC Score : {roc_auc:.6f}")
    logger.info("Matrice de confusion :\n%s", confusion_matrix(y_test, y_pred_labels))
    logger.info(
        f"Classe 1 - Precision: {class1_metrics['precision']:.4f}, Recall: {class1_metrics['recall']:.4f}, F1: {class1_metrics['f1-score']:.4f}"
    )

    false_negatives = np.sum((y_test == 0) & (y_pred_labels == 1))
    total_errors = np.sum(y_test == 0)
    fn_cost = false_negatives / total_errors if total_errors > 0 else 0

    logger.info(f"Classe 0 (Erreurs) - Precision: {class0_report['precision']:.4f}, "
                f"Recall: {class0_report['recall']:.4f}, "
                f"F1: {class0_report['f1-score']:.4f}")
    logger.info(f"Faux Négatifs (Erreurs non détectées): {false_negatives}/{total_errors} = {fn_cost:.2%}")

    # === Analyse feature importance ===
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_sum = shap_df.abs().mean().sort_values(ascending=False)
    logger.info("Top 10 des features par importance SHAP:")
    for feat, imp in shap_sum.head(30).items():
        logger.info(f"  {feat}: {imp:.4f}")

    shap_path = os.path.join(project_root, 'meta_model', 'results', f'seed_{seed}')
    os.makedirs(shap_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_path, 'shap_feature_importance.png'))
    plt.close()

    # === Sauvegarde du modèle ===
    results_dir = os.path.join(project_root, 'meta_model', 'results', f'seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"xgboost_meta_model_seed_{seed}.joblib")
    joblib.dump({
        'model': calibrated_model,
        'threshold': float(best_threshold),
        'class1_precision': float(class1_metrics['precision']),
        'class1_recall': float(class1_metrics['recall'])
    }, model_path)
    logger.info(f"Modèle sauvegardé : {model_path}")

    # === Sauvegarde des résultats et visualisations ===
    fpr, tpr, _ = roc_curve((y_test == 0).astype(int), y_prob_class0)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbe ROC (Seed {seed})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(y_prob_class0[y_test == 1], bins=50, alpha=0.5, label='Correct (Classe 1)')
    plt.hist(y_prob_class0[y_test == 0], bins=50, alpha=0.5, label='Erreur (Classe 0)')
    plt.axvline(x=best_threshold, linestyle='--', label=f'Seuil: {best_threshold:.2f}')
    plt.title('Distribution des Probabilités Prédites (proba Erreur = classe 0)')
    plt.xlabel('Probabilité Erreur (classe 0)')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'probability_distribution.png'))
    plt.close()

    # Sauvegarde des prédictions
    np.save(os.path.join(results_dir, 'xgboost_meta_model_probs.npy'), y_prob_class0)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_y_true.npy'), y_true_test)
    np.save(os.path.join(results_dir, 'exo_model_y_pred.npy'), y_pred_exo_test)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_y_pred.npy'), y_pred_labels)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_X_test.npy'), X_test.values)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_time_test.npy'), time_test)

    with open(os.path.join(results_dir, 'model_params.json'), 'w') as f:
        json.dump({
            'optimal_threshold': float(best_threshold),
            'roc_auc': float(roc_auc),
            'class_distribution': {int(k): float(v) for k, v in class_dist.to_dict().items()},
            'class1_precision': float(class1_metrics['precision']),
            'class1_recall': float(class1_metrics['recall']),
            'beta_for_errors': float(BETA_FOR_ERRORS),
            'min_precision_error': float(MIN_PREC0)
        }, f, indent=4)

    return {
        'roc_auc': roc_auc,
        'class1_precision': class1_metrics['precision'],
        'class1_recall': class1_metrics['recall'],
        'meta_class_dist_0': float(class_dist[0]),
        'meta_class_dist_1': float(class_dist[1]),
        'meta_f1_score': float(class1_metrics['f1-score'])
    }


if __name__ == "__main__":
    from exogenous_model_v0.dataset.generate_dataset import logger
    train_and_test_meta_xgboost(42, logger)
