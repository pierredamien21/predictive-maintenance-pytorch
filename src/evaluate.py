"""Métriques, rechargement d'un modèle entraîné et importance par permutation."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .data import CMAPSSDataset, load_test, prepare, windows_for
from .models import build_model


def nasa_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Score asymétrique PHM08 : une prédiction trop optimiste (d > 0) coûte plus cher."""
    d = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """RMSE, MAE (en cycles) et score NASA."""
    d = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(d**2))),
        "mae": float(np.mean(np.abs(d))),
        "nasa": nasa_score(targets, predictions),
    }


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """F1 macro, rappel de la classe critique (2) et matrice de confusion."""
    labels = [0, 1, 2]
    return {
        "f1_macro": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "recall_critical": float(
            recall_score(targets, predictions, labels=[2], average="macro", zero_division=0)
        ),
        "confusion": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module, windows: dict, rul_max: int, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray | None]:
    """RUL prédite en cycles (bornée à [0, rul_max]) et classe prédite (ou None)."""
    model.eval()
    dataset = CMAPSSDataset(windows, rul_max)
    ruls, states = [], []
    for start in range(0, len(dataset), 2048):
        out = model(
            dataset.seq[start : start + 2048].to(device),
            dataset.ctx[start : start + 2048].to(device),
        )
        rul, logits = out if isinstance(out, tuple) else (out, None)
        ruls.append(rul.cpu().numpy())
        if logits is not None:
            states.append(logits.argmax(dim=1).cpu().numpy())
    rul = np.clip(np.concatenate(ruls) * rul_max, 0, rul_max)
    return rul, (np.concatenate(states) if states else None)


def load_run(
    name: str, seed: int = 0, results_dir: str = "results/runs", models_dir: str = "models"
) -> tuple:
    """Recharge un modèle entraîné : renvoie `(model, bundle, result)`.

    `bundle` est le pipeline de données refait à l'identique (même split, même scaler).
    """
    result = json.loads((Path(results_dir) / f"{name}_seed{seed}.json").read_text(encoding="utf-8"))
    cfg = result["config"]
    bundle = prepare(cfg["data"], cfg["seed"])
    model = build_model(
        cfg["model"], len(bundle["features"]), cfg["data"]["window"], bundle["scaler"].n_regimes
    )
    model.load_state_dict(
        torch.load(
            Path(models_dir) / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=True
        )
    )
    return model.eval(), bundle, result


def bootstrap_rmse(
    targets: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """RMSE et intervalle de confiance à 95 % par rééchantillonnage des moteurs.

    `preds` : `(n_moteurs,)` ou `(n_graines, n_moteurs)` ; dans ce cas la RMSE est moyennée
    sur les graines. Avec `preds_b`, renvoie l'écart apparié `RMSE(a) - RMSE(b)` : si
    l'intervalle ne contient pas 0, l'écart ne s'explique pas par le tirage des moteurs.
    """
    targets = np.asarray(targets, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(targets), (n_boot, len(targets)))
    idx = np.vstack([np.arange(len(targets)), idx])  # ligne 0 : l'échantillon d'origine

    def rmse(preds):
        sq = (np.atleast_2d(np.asarray(preds, dtype=float)) - targets) ** 2
        return np.sqrt(sq[:, idx].mean(axis=-1)).mean(axis=0)

    values = rmse(preds_a) - (rmse(preds_b) if preds_b is not None else 0)
    low, high = np.percentile(values[1:], [2.5, 97.5])
    return float(values[0]), float(low), float(high)


def test_predictions(name: str, seed: int = 0) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Prédictions sur la dernière fenêtre de chaque moteur du jeu de test officiel."""
    model, bundle, _ = load_run(name, seed)
    cfg = bundle["cfg"]
    windows = windows_for(load_test(cfg), bundle["scaler"], cfg, last_only=True)
    return (windows, *predict(model, windows, cfg["rul_max"]))


def permutation_importance(
    model, windows: dict, features: list[str], rul_max: int, seed: int = 0
) -> dict[str, float]:
    """Hausse de RMSE quand on mélange un capteur entre les fenêtres (haut = capteur important)."""
    rng = np.random.default_rng(seed)
    base = regression_metrics(windows["rul"], predict(model, windows, rul_max)[0])["rmse"]
    gains = {}
    for i, feature in enumerate(features):
        shuffled = dict(windows, X=windows["X"].copy())
        shuffled["X"][:, :, i] = shuffled["X"][rng.permutation(len(shuffled["X"])), :, i]
        gains[feature] = (
            regression_metrics(windows["rul"], predict(model, shuffled, rul_max)[0])["rmse"] - base
        )
    return gains
