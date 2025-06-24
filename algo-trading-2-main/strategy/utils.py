
def compute_capture_ratio(y_true, y_pred, y_proba, mode='weighted', threshold=0.6, gain=20, loss=7):
    """
    Calcule le capture ratio selon le mode choisi.

    Args:
        y_true (array): Valeurs réelles.
        y_pred (array): Prédictions (1=BUY, 2=SELL, 0=HOLD).
        y_proba (array): Probabilités associées.
        mode (str): 'with_proba' ou 'weighted'.
        threshold (float): Seulement pour 'with_proba' (filtrer les signaux faibles).
        gain (float): Gain en cas de bon trade.
        loss (float): Perte en cas de mauvais trade.

    Returns:
        float: capture ratio.
    """
    if mode == 'with_proba':
        # Mode qui ne compte que les signaux avec proba >= threshold
        model_profit = 0
        oracle_profit = 0

        for true, pred, proba in zip(y_true, y_pred, y_proba):
            if proba < threshold or pred == 0:
                continue
            if pred == 1:
                model_profit += gain if true == 1 else -loss
            elif pred == 2:
                model_profit += gain if true == 2 else -loss
            if true in (1, 2):
                oracle_profit += gain

        if oracle_profit == 0:
            return 0
        return model_profit / oracle_profit

    elif mode == 'weighted':
        # Mode pondéré par les probabilités (aucun seuil)
        model_profit = 0
        oracle_profit = 0

        for true, pred, proba in zip(y_true, y_pred, y_proba):
            weight = proba
            if pred == 1:
                model_profit += weight * (gain if true == 1 else -loss)
            elif pred == 2:
                model_profit += weight * (gain if true == 2 else -loss)
            if true in (1, 2):
                oracle_profit += gain

        if oracle_profit == 0:
            return 0
        return model_profit / oracle_profit

    else:
        raise ValueError(f"Mode inconnu : {mode}. Choisir 'with_proba' ou 'weighted'.")


def analyse_capture_ratio(seed) :

    y_proba_path = r'../meta_model/train_and_test\xgboost_meta_model_probs.npy'
    y_pred_path = r'../meta_model/train_and_test\xgboost_meta_model_y_pred.npy'
    y_true_path = r'../meta_model/train_and_test\xgboost_meta_model_y_true.npy'

    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path)
    y_proba = np.load(y_proba_path)

    # === Calcul des capture ratios pour différents seuils ===
    capture_ratio = []
    coverage = []
    mode = 'with_proba'  # 'weigthed ou 'with_proba'

    for threshold in range(50, 100, 5):
        t = threshold / 100
        mask = y_proba >= t
        if mask.sum() > 0:
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]
            y_proba_masked = y_proba[mask]

            if mode == 'with_proba':
                cap = compute_capture_ratio(y_true_masked, y_pred_masked, y_proba_masked, mode=mode, threshold=t)
            else:
                cap = compute_capture_ratio(y_true_masked, y_pred_masked, y_proba_masked, mode=mode)

            capture_ratio.append(cap)
            coverage.append(mask.sum() / len(y_true))
        else:
            capture_ratio.append(0)
            coverage.append(0)

    thresholds = [t / 100 for t in range(50, 100, 5)]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(thresholds, capture_ratio, marker='o', color='blue', label="Capture Ratio")
    ax1.set_ylabel("Capture Ratio", color='blue')
    ax1.set_xlabel("Seuil de probabilité")
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Deuxième axe pour coverage
    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverage, marker='s', color='green', label="% de trades retenus")
    ax2.set_ylabel("% de trades retenus", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title("🎯 Capture Ratio vs Seuil de Confiance avec Couverture")
    plt.xticks(thresholds)
    plt.tight_layout()
    plt.show()
