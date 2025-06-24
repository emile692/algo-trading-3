"""train_test_meta_xgboost.py"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


def train_and_test_meta_xgboost(seed):

    # === Chargement du dataset méta ===
    df = pd.read_csv(f"C:/Donnees/ECL/11-S9/Deep Learning/algo-trading-2/meta_model/dataset/features_and_target/meta_dataset_seed_{seed}.csv")

    # === Target : prédiction correcte ou non ===
    df["meta_label"] = df["y_true"] == df["y_pred"]
    df["meta_label"] = df["meta_label"].astype(int)

    # === Séparation X / y ===
    drop_cols = ["y_true", "y_pred", "meta_label"]
    X = df.drop(columns=drop_cols)
    y = df["meta_label"]

    y_true_full = df["y_true"].to_numpy()
    y_pred_full = df["y_pred"].to_numpy()

    # === Split train_and_test / test (avec stratify sur meta_label) ===
    X_train, X_test, y_train, y_test, y_true_train, y_true_test, y_pred_train, y_pred_test = train_test_split(
        X, y, y_true_full, y_pred_full, stratify=y, test_size=0.2, random_state=42
    )

    # === Entraînement du modèle XGBoost ===
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    print("[•] Entraînement du méta-modèle...")
    model.fit(X_train, y_train)

    # === Évaluation ===
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n[✓] Évaluation sur le set de test :")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score :", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))

    # === Sauvegarde du modèle ===
    joblib.dump(model, "xgboost_meta_model.joblib")
    print("\n[✓] Modèle sauvegardé : xgboost_meta_model.joblib")

    np.save(f"results/seed_{seed}/xgboost_meta_model_probs.npy", y_prob)
    np.save(f"results/seed_{seed}/xgboost_meta_model_y_true.npy", y_true_test)
    np.save(f"results/seed_{seed}/xgboost_meta_model_y_pred.npy", y_pred_test)
    print(f"\n[✓] Résultas sauvegardés dans : results/seed_{seed}/xgboost_meta_model_probs.npy")
