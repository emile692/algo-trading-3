# analyze_errors.py  — méthode 2 (alignement par time via artefacts méta) + visualisations Entropy
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.feature_selection import mutual_info_classif


def try_load_model_predict_proba(model_path: str, X: pd.DataFrame):
    """
    Option secondaire : si les .npy n'existent pas, on tente de recalculer
    les probas à partir du modèle sauvegardé.
    Retourne (y_pred_prob_correct, y_pred, threshold) ou (None, None, None) si impossible.
    """
    try:
        obj = joblib.load(model_path)
    except Exception:
        return None, None, None

    clf = None
    meta = {}

    if hasattr(obj, "predict_proba"):
        clf = obj
    elif isinstance(obj, dict):
        meta = obj
        for k in ["calibrated_model", "model", "clf", "estimator", "xgb", "best_estimator_"]:
            if k in obj and hasattr(obj[k], "predict_proba"):
                clf = obj[k]
                break

    if clf is None:
        return None, None, None

    try:
        y_pred_prob_class0 = clf.predict_proba(X)[:, 0]
        y_pred_prob_correct = 1.0 - y_pred_prob_class0  # NEW: cohérence avec P(correct)
        thr = meta.get("threshold", meta.get("optimal_threshold", 0.5)) if isinstance(meta, dict) else 0.5
        y_pred = (y_pred_prob_class0 >= thr).astype(int)  # 1=prédit erreur -> labels {0,1} = {erreur, correct}
        y_pred = np.where(y_pred == 1, 0, 1)             # NEW: convertir en labels {0=erreur,1=correct}
        return y_pred_prob_correct, y_pred, thr
    except Exception:
        return None, None, None


def safe_histplot(df, col, hue, path, title, bins=30):
    if col not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    try:
        sns.histplot(data=df, x=col, hue=hue, kde=True, bins=bins, stat="density", common_norm=False)
    except Exception:
        for lab, sub in df.groupby(hue):
            sns.histplot(sub[col].dropna(), label=str(lab), kde=True, bins=bins, stat="density")
        plt.legend(title=hue)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def error_rate_by_quantile(df, value_col, bins=10):
    if value_col not in df.columns:
        return None
    tmp = df[[value_col, "is_error"]].dropna().copy()
    tmp["q"] = pd.qcut(tmp[value_col], q=bins, duplicates="drop")
    grp = tmp.groupby("q", observed=True)["is_error"].mean().reset_index()
    grp["bin_center"] = tmp.groupby("q", observed=True)[value_col].mean().values
    grp.rename(columns={"is_error": "error_rate"}, inplace=True)
    return grp


def heatmap_2d_error(df, x_col, y_col, x_bins=20, y_bins=20):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    sub = df[[x_col, y_col, "is_error"]].dropna().copy()
    sub["x_bin"] = pd.qcut(sub[x_col], q=min(x_bins, sub[x_col].nunique()), duplicates="drop")
    sub["y_bin"] = pd.qcut(sub[y_col], q=min(y_bins, sub[y_col].nunique()), duplicates="drop")
    piv = sub.groupby(["y_bin", "x_bin"], observed=True)["is_error"].mean().unstack()
    return piv


def main(seed: int):
    # ---------- chemins ----------
    project_root = os.path.abspath(os.path.dirname(__file__))
    dataset_csv = os.path.join(
        project_root, "meta_model", "dataset", "features_and_target", f"meta_dataset_seed_{seed}.csv"
    )
    results_dir = os.path.join(project_root, "meta_model", "results", f"seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"xgboost_meta_model_seed_{seed}.joblib")
    # CHG: on supprime la dépendance au trimming heuristique et on aligne par 'time'
    probs_npy = os.path.join(results_dir, "xgboost_meta_model_probs.npy")           # prob_class0
    ypred_npy = os.path.join(results_dir, "xgboost_meta_model_y_pred.npy")          # labels {0,1}
    time_npy  = os.path.join(results_dir, "xgboost_meta_model_time_test.npy")       # NEW: temps xgboost test

    print(f"[INFO] Chargement dataset: {dataset_csv}")
    df_full = pd.read_csv(dataset_csv)

    if "meta_label" not in df_full.columns:
        raise ValueError("La colonne 'meta_label' est absente du dataset méta.")

    # ---------- ALIGNEMENT PAR TIME (robuste) ----------
    if os.path.exists(probs_npy) and os.path.exists(ypred_npy) and os.path.exists(time_npy):
        print("[INFO] Alignement par 'time' via artefacts méta (.npy).")
        prob_class0 = np.load(probs_npy)
        y_pred_lbls = np.load(ypred_npy)
        time_test = np.load(time_npy, allow_pickle=True)


        if len(prob_class0) != len(y_pred_lbls) or len(prob_class0) != len(time_test):
            raise ValueError("Mismatch tailles entre probs, y_pred et time test (méta).")

        # Construire un DF prédictions + temps
        pred_df = pd.DataFrame({
            "time": time_test,
            "y_pred_prob_class0": prob_class0,
            "y_pred_meta": y_pred_lbls
        })

        # Harmoniser le type de time pour le merge
        # Si 'time' du CSV est string ISO, on normalise les deux côtés en string
        # (on évite les soucis tz/np.datetime64)
        pred_df["time"] = pred_df["time"].astype(str)
        df_full["time"] = df_full["time"].astype(str)

        # Merge INNER: on ne garde que les points réellement scorés par le méta
        df = df_full.merge(pred_df, on="time", how="inner").reset_index(drop=True)

        # Convertir la proba classe0 (erreur) en P(correct) pour cohérence avec visus existants
        df["y_pred_prob"] = 1.0 - df["y_pred_prob_class0"]
        y = df["meta_label"].astype(int)
        y_pred = df["y_pred_meta"].astype(int)
        threshold_used = None
        # Essai lecture seuil depuis le .joblib
        try:
            obj = joblib.load(model_path)
            if isinstance(obj, dict) and "threshold" in obj:
                threshold_used = float(obj["threshold"])
        except Exception:
            pass
        if threshold_used is None:
            threshold_used = 0.5

        # Définir X pour les analyses (on exclut colonnes cibles et techniques)
        drop_cols = [c for c in ["meta_label", "y_true", "y_pred", "time",
                                 "y_pred_prob_class0", "y_pred_meta", "y_pred_prob"] if c in df.columns]
        X = df.drop(columns=drop_cols)

    else:
        print("[WARN] Artefacts .npy manquants — fallback au modèle .joblib (moins robuste).")
        df = df_full.copy()
        y = df["meta_label"].astype(int)
        drop_cols = [c for c in ["meta_label", "y_true", "y_pred"] if c in df.columns]
        X = df.drop(columns=drop_cols)

        y_pred_prob, y_pred, threshold_used = try_load_model_predict_proba(model_path, X)
        if y_pred_prob is None or y_pred is None:
            raise RuntimeError("Ni artefacts .npy présents, ni modèle exploitable pour predict_proba.")

        df["y_pred_prob"] = y_pred_prob

    # ---------- Sanity checks ----------
    assert len(y) == len(df), "Taille y != df"
    assert len(y_pred) == len(df), "Taille y_pred != df"

    # ---------- Stats globales ----------
    print("\n[INFO] Statistiques globales du méta-modèle :")
    print(classification_report(y, y_pred, digits=3))
    try:
        roc = roc_auc_score(y, df["y_pred_prob"])
        print(f"ROC AUC : {roc:.6f}")
    except Exception:
        roc = float("nan")

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=["Erreur (0)", "Correct (1)"], columns=["Prédit Erreur", "Prédit Correct"])
    print("\nMatrice de confusion :\n", cm_df)

    # ---------- Analyses erreurs (par rapport aux erreurs LSTM) ----------
    df["is_error"] = (df["meta_label"] == 0).astype(int)

    # Corrélation Spearman avec is_error
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    spearmans = []
    for col in numeric_cols:
        try:
            corr, p = spearmanr(X[col], df["is_error"])
            spearmans.append((col, corr, p))
        except Exception:
            continue
    spearman_df = pd.DataFrame(spearmans, columns=["feature", "spearman_corr", "p_value"]).dropna()
    spearman_df["abs_corr"] = spearman_df["spearman_corr"].abs()
    spearman_df.sort_values("abs_corr", ascending=False, inplace=True)
    spearman_df.to_csv(os.path.join(results_dir, "feature_error_spearman.csv"), index=False)

    # Mutual Information (non-linéaire)
    mi = mutual_info_classif(X[numeric_cols].fillna(0), df["is_error"], discrete_features=False, random_state=seed)
    mi_df = pd.DataFrame({"feature": numeric_cols, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    mi_df.to_csv(os.path.join(results_dir, "feature_error_mutual_info.csv"), index=False)

    print("\n[INFO] Top 10 Spearman (|corr|) :")
    print(spearman_df.head(10))
    print("\n[INFO] Top 10 Mutual Information :")
    print(mi_df.head(10))

    # ---------- Graphiques généraux ----------
    sns.set(style="whitegrid", font_scale=1.05)

    # 1) Top corr Spearman
    top_corr = spearman_df.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_corr, x="abs_corr", y="feature")
    plt.title(f"Top 15 corrélations (|Spearman|) avec erreurs LSTM - seed {seed}")
    plt.xlabel("|Spearman| vs is_error")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "top_corr_features.png"))
    plt.close()

    # 2) Top MI
    top_mi = mi_df.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_mi, x="mutual_info", y="feature")
    plt.title(f"Top 15 Mutual Information avec erreurs LSTM - seed {seed}")
    plt.xlabel("Mutual Information")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "top_mi_features.png"))
    plt.close()

    # 3) Heatmap corr avec is_error
    try:
        corr_matrix = pd.concat([df["is_error"], X[numeric_cols]], axis=1).corr(method="spearman")
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
        plt.title(f"Heatmap corr (Spearman) avec is_error - seed {seed}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "error_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"[WARN] Heatmap non générée: {e}")

    # 4) Histogramme des probas du méta (P(correct))
    plt.figure(figsize=(9, 5))
    sns.histplot(df.loc[y == 1, "y_pred_prob"], label="Correct (1)", kde=True)
    sns.histplot(df.loc[y == 0, "y_pred_prob"], label="Erreur (0)", kde=True)
    plt.legend()
    plt.title(f"Distribution des probas du méta-modèle - seed {seed} (thr={threshold_used:.3f})")
    plt.xlabel("P(correct)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "probability_distribution.png"))
    plt.close()

    # 5) ROC
    try:
        fpr, tpr, _ = roc_curve(y, df["y_pred_prob"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC méta-modèle - seed {seed}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "roc_curve.png"))
        plt.close()
    except Exception:
        pass

    # 6) Matrice de confusion en heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matrice de confusion méta - seed {seed}")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # ---------- VISU spécifiques Entropy ----------
    ent_cols = [c for c in ["entropy_price", "entropy_pred", "entropy_pred_roll"] if c in df.columns]

    # A) Distributions des entropies selon is_error
    for c in ent_cols:
        safe_histplot(
            df, c, "is_error",
            os.path.join(results_dir, f"{c}_hist_by_is_error.png"),
            title=f"Distribution de {c} selon is_error - seed {seed}",
            bins=40
        )

    # B) Taux d'erreur par quantiles d'entropie (courbes)
    for c in ent_cols:
        grp = error_rate_by_quantile(df, c, bins=12)
        if grp is not None and len(grp) > 0:
            plt.figure(figsize=(9, 5))
            plt.plot(grp["bin_center"], grp["error_rate"], marker="o")
            plt.title(f"Taux d'erreur vs {c} (par quantiles) - seed {seed}")
            plt.xlabel(c)
            plt.ylabel("Error rate (mean is_error)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"error_rate_vs_{c}.png"))
            plt.close()

    # C) Heatmap 2D : error rate en fct de (entropy_price, entropy_pred)
    if "entropy_price" in df.columns and "entropy_pred" in df.columns:
        piv = heatmap_2d_error(df, "entropy_price", "entropy_pred", x_bins=18, y_bins=18)
        if piv is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(piv, cmap="mako", cbar_kws={"label": "Error rate"})
            plt.title(f"Taux d'erreur 2D — entropy_pred (Y) vs entropy_price (X) - seed {seed}")
            plt.xlabel("entropy_price (bins)")
            plt.ylabel("entropy_pred (bins)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "error_rate_heatmap_entropy2D.png"))
            plt.close()

    # D) Calibration par incertitude (P(correct) vs entropy_pred quantiles)
    if "entropy_pred" in df.columns and "y_pred_prob" in df.columns:
        tmp = df[["entropy_pred", "y_pred_prob", "is_error"]].dropna().copy()
        tmp["q"] = pd.qcut(tmp["entropy_pred"], q=10, duplicates="drop")
        calib = tmp.groupby("q", observed=True).agg(
            mean_entropy=("entropy_pred", "mean"),
            mean_p_correct=("y_pred_prob", "mean"),
            error_rate=("is_error", "mean")
        ).reset_index(drop=False)
        calib.to_csv(os.path.join(results_dir, "calibration_by_entropy_pred.csv"), index=False)

        plt.figure(figsize=(9, 5))
        plt.plot(calib["mean_entropy"], calib["mean_p_correct"], marker="o")
        plt.title(f"Calibration: E[P(correct)] vs entropy_pred (quantiles) - seed {seed}")
        plt.xlabel("entropy_pred (mean per quantile)")
        plt.ylabel("Mean predicted P(correct)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "calibration_vs_entropy_pred.png"))
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(calib["mean_entropy"], calib["error_rate"], marker="o")
        plt.title(f"Error rate vs entropy_pred (quantiles) - seed {seed}")
        plt.xlabel("entropy_pred (mean per quantile)")
        plt.ylabel("Error rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "error_rate_vs_entropy_pred_quantiles.png"))
        plt.close()

    # E) Séries temporelles (si 'time' dispo) : entropies + rolling error-rate
    if "time" in df.columns:
        try:
            dfts = df.copy()
            dfts["time"] = pd.to_datetime(dfts["time"], errors="coerce")
            dfts = dfts.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            dfts["err_roll"] = dfts["is_error"].rolling(200, min_periods=50).mean()

            plt.figure(figsize=(12, 6))
            if "entropy_price" in dfts.columns:
                plt.plot(dfts["time"], dfts["entropy_price"], label="entropy_price")
            if "entropy_pred" in dfts.columns:
                plt.plot(dfts["time"], dfts["entropy_pred"], alpha=0.7, label="entropy_pred")
            if "entropy_pred_roll" in dfts.columns:
                plt.plot(dfts["time"], dfts["entropy_pred_roll"], alpha=0.7, label="entropy_pred_roll")
            plt.plot(dfts["time"], dfts["err_roll"], label="error_rate (rolling)", linewidth=2)
            plt.legend()
            plt.title(f"Séries temporelles : entropies & rolling error-rate - seed {seed}")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "timeseries_entropy_vs_error.png"))
            plt.close()
        except Exception as e:
            print(f"[WARN] Time-series entropy plot non généré: {e}")

    # 7) Mini récap JSON
    recap = {
        "seed": seed,
        "n_rows": int(len(df)),
        "roc_auc": float(roc) if roc == roc else None,
        "threshold_used": float(threshold_used),
        "class_report": classification_report(y, y_pred, digits=3, output_dict=True),
    }
    with open(os.path.join(results_dir, "model_params.json"), "w", encoding="utf-8") as f:
        json.dump(recap, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analyse terminée — résultats dans {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
