"""train_test_meta_xgboost.py"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.utils import class_weight


def train_and_test_meta_xgboost(seed, logger):
    # === Chargement du dataset méta ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    meta_dataset_path = os.path.join(project_root, 'meta_model', 'dataset', 'features_and_target',
                                     f"meta_dataset_seed_{seed}.csv")
    df = pd.read_csv(meta_dataset_path)

    # Analyse du déséquilibre de classes
    class_dist = df["meta_label"].value_counts(normalize=True)
    logger.info(
        f"Distribution des classes méta: Classe 0 (erreur): {class_dist[0]:.2%}, Classe 1 (correct): {class_dist[1]:.2%}")

    # Calcul des poids de classe - stratégie métier
    class_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=df['meta_label']
    )
    sample_weights = pd.Series(class_weights, index=df.index)

    # === Séparation X / y ===
    drop_cols = ["y_true", "y_pred", "meta_label"]
    X = df.drop(columns=drop_cols)
    y = df["meta_label"]
    y_true_full = df["y_true"].to_numpy()
    y_pred_full = df["y_pred"].to_numpy()

    # === Split train / test (avec stratify sur meta_label) ===
    (X_train, X_test,
     y_train, y_test,
     sw_train, sw_test,
     y_true_train, y_true_test,
     y_pred_train, y_pred_test) = train_test_split(
        X, y, sample_weights, y_true_full, y_pred_full,
        stratify=y, test_size=0.2, random_state=seed
    )

    # === Validation croisée pour l'optimisation du seuil ===
    logger.info("Optimisation du seuil de décision par validation croisée...")
    thresholds = []
    best_threshold = 0.5

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Réinitialiser le modèle pour chaque fold
        fold_model = XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=seed,  # Différentes graines
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

        y_prob_val = fold_model.predict_proba(X_fold_val)[:, 1]
        fpr, tpr, t = roc_curve(y_fold_val, y_prob_val)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = t[optimal_idx]
        thresholds.append(optimal_threshold)
        logger.info(f"Fold {fold + 1}: seuil optimal = {optimal_threshold:.4f}")

    best_threshold = np.median(thresholds)
    logger.info(f"Seuil optimal médian: {best_threshold:.4f}")

    # === Entraînement final avec calibration ===
    logger.info("Entraînement final du méta-modèle...")
    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )

    final_model.fit(X_train, y_train, sample_weight=sw_train)

    logger.info("Calibration du modèle...")
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic')
    calibrated_model.fit(X_train, y_train)

    # === Évaluation avec seuil optimisé ===
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > best_threshold).astype(int)

    # Métriques complètes
    logger.info("Évaluation sur le set de test :")
    logger.info(classification_report(y_test, y_pred))

    # Focus sur la classe 1 (correct predictions)
    class1_report = classification_report(
        y_test,
        y_pred,
        target_names=["Erreur (0)", "Correct (1)"],
        output_dict=True
    )
    class1_metrics = class1_report["Correct (1)"]

    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC AUC Score : {roc_auc:.6f}")
    logger.info("Matrice de confusion :\n%s", confusion_matrix(y_test, y_pred))
    logger.info(
        f"Classe 1 - Precision: {class1_metrics['precision']:.4f}, Recall: {class1_metrics['recall']:.4f}, F1: {class1_metrics['f1-score']:.4f}")

    # === Analyse feature importance ===
    feature_importance = final_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    logger.info("Top 10 des features importantes:")
    for i in sorted_idx[:10]:
        logger.info(f"  {X.columns[i]}: {feature_importance[i]:.4f}")

    # === Sauvegarde du modèle ===
    results_dir = os.path.join(project_root, 'meta_model', 'results', f'seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"xgboost_meta_model_seed_{seed}.joblib")
    # Sauvegarder modèle + seuil
    joblib.dump({
        'model': calibrated_model,
        'threshold': best_threshold,
        'class1_precision': class1_metrics['precision'],
        'class1_recall': class1_metrics['recall']
    }, model_path)
    logger.info(f"Modèle sauvegardé : {model_path}")

    # === Sauvegarde des résultats et visualisations ===
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
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

    # Distribution des probabilités
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='Correct (Classe 1)', color='green')
    plt.hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='Erreur (Classe 0)', color='red')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Seuil: {best_threshold:.2f}')
    plt.title('Distribution des Probabilités Prédites')
    plt.xlabel('Probabilité Prédite (Classe 1)')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'probability_distribution.png'))
    plt.close()

    # Sauvegarde des prédictions
    np.save(os.path.join(results_dir, 'xgboost_meta_model_probs.npy'), y_prob)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_y_true.npy'), y_true_test)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_y_pred.npy'), y_pred_test)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_X_test.npy'), X_test.values)

    # Paramètres du modèle
    with open(os.path.join(results_dir, 'model_params.json'), 'w') as f:
        json.dump({
            'optimal_threshold': float(best_threshold),
            'roc_auc': float(roc_auc),
            'class_distribution': {int(k): float(v) for k, v in class_dist.to_dict().items()},
            'class1_precision': float(class1_metrics['precision']),
            'class1_recall': float(class1_metrics['recall']),
            'feature_importances': {
                X.columns[i]: float(feature_importance[i])
                for i in sorted_idx[:10]
            }
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

    from exogenous_model.dataset.generate_dataset import logger
    train_and_test_meta_xgboost(42, logger)