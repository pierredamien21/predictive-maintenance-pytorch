"""Baselines classiques : moyenne, régression linéaire, Random Forest, gradient boosting.

python -m src.baselines    # règle le Random Forest sur la validation de FD001 et FD004
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .evaluate import regression_metrics

FOREST_PARAMS_PATH = Path("results/forest_params.json")
DEFAULT_FOREST = {"min_samples_leaf": 5, "max_features": 1.0}
FOREST = "Random Forest (features de fenêtre)"
BOOSTING = "Gradient boosting (features de fenêtre)"


def window_features(X: np.ndarray) -> np.ndarray:
    """Résume chaque fenêtre `(n, W, F)` : moyenne, écart-type, pente, dernière valeur."""
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2  # temps centré
    slope = np.einsum("w,nwf->nf", t, X) / np.sum(t**2)  # pente des moindres carrés
    return np.concatenate([X.mean(axis=1), X.std(axis=1), slope, X[:, -1, :]], axis=1)


def forest_params(subset: str) -> dict:
    """Hyperparamètres du Random Forest réglés sur la validation (valeurs par défaut sinon)."""
    if FOREST_PARAMS_PATH.exists():
        tuned = json.loads(FOREST_PARAMS_PATH.read_text(encoding="utf-8"))
        return tuned.get(subset, DEFAULT_FOREST)
    return DEFAULT_FOREST


def fit_forest(train: dict, params: dict, seed: int = 0) -> RandomForestRegressor:
    forest = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed, **params)
    return forest.fit(window_features(train["X"]), train["rul"])


def tune_forest(train: dict, val: dict, rul_max: int = 125) -> pd.DataFrame:
    """Grille sur `min_samples_leaf` et `max_features`, évaluée sur la validation."""
    rows = []
    for leaf in [1, 5, 15]:
        for max_features in ["sqrt", 0.33, 1.0]:
            params = {"min_samples_leaf": leaf, "max_features": max_features}
            pred = fit_forest(train, params).predict(window_features(val["X"]))
            rmse = regression_metrics(val["rul"], np.clip(pred, 0, rul_max))["rmse"]
            rows.append({**params, "rmse": rmse})
    return pd.DataFrame(rows).sort_values("rmse", ignore_index=True)


def run_baselines(
    train: dict,
    eval_sets: dict[str, dict],
    rul_max: int = 125,
    seed: int = 0,
    forest: dict | None = None,
) -> pd.DataFrame:
    """Ajuste les baselines sur `train` et les évalue sur chaque jeu de `eval_sets`.

    Renvoie une ligne par (modèle, jeu) avec les métriques et les prédictions (`pred`).
    """
    features = window_features(train["X"])
    rf = fit_forest(train, forest or DEFAULT_FOREST, seed)
    boosting = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    boosting.fit(features, train["rul"])
    linear = LinearRegression().fit(train["X"][:, -1, :], train["rul"])
    predictors = {
        "Moyenne du train": lambda w: np.full(len(w["rul"]), train["rul"].mean()),
        "Régression linéaire (dernier cycle)": lambda w: linear.predict(w["X"][:, -1, :]),
        FOREST: lambda w: rf.predict(window_features(w["X"])),
        BOOSTING: lambda w: boosting.predict(window_features(w["X"])),
    }
    rows = []
    for name, fn in predictors.items():
        for set_name, w in eval_sets.items():
            pred = np.clip(fn(w), 0, rul_max)
            metrics = regression_metrics(w["rul"], pred)
            rows.append({"modèle": name, "jeu": set_name, **metrics, "pred": pred})
    return pd.DataFrame(rows)


def main() -> None:
    from .data import prepare
    from .experiments import EXPERIMENTS
    from .train import apply_overrides
    from .utils import load_config

    best, tables = {}, []
    for subset, name in [("FD001", "gru_best_fd001"), ("FD004", "multitask_w40_fd004")]:
        _, base, overrides = EXPERIMENTS[name]  # mêmes données que le réseau retenu
        cfg = apply_overrides(load_config(base), overrides)
        bundle = prepare(cfg["data"], cfg["seed"])
        table = tune_forest(bundle["train"], bundle["val"], cfg["data"]["rul_max"])
        print(subset, table.round(2).to_string(), sep="\n")
        leaf, max_features = table.loc[0, ["min_samples_leaf", "max_features"]]
        best[subset] = {
            "min_samples_leaf": int(leaf),
            "max_features": max_features if max_features == "sqrt" else float(max_features),
        }
        tables.append(table.assign(jeu=subset))
    FOREST_PARAMS_PATH.write_text(json.dumps(best, indent=1), encoding="utf-8")
    pd.concat(tables).to_csv("results/forest_tuning.csv", index=False)


if __name__ == "__main__":
    main()
