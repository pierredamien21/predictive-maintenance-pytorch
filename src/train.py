"""Boucle d'entraînement écrite à la main, commune à tous les modèles.

python -m src.train --config configs/multitask.yaml --seeds 0 1 2 3 4
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CMAPSSDataset, prepare
from .evaluate import classification_metrics, predict, regression_metrics
from .models import build_model, count_parameters
from .utils import load_config, save_checkpoint, set_seed


def combined_loss(
    out, rul, state, lam: float, ce: nn.CrossEntropyLoss
) -> tuple[torch.Tensor, float, float]:
    """Perte totale et ses deux composantes. Mono-sortie : MSE seule."""
    if not isinstance(out, tuple):
        mse = nn.functional.mse_loss(out, rul)
        return mse, mse.item(), 0.0
    rul_pred, logits = out
    mse, cross = nn.functional.mse_loss(rul_pred, rul), ce(logits, state)
    return lam * mse + (1 - lam) * cross, mse.item(), cross.item()


def run_epoch(model, loader, device, lam, ce, optimizer=None, clip_norm=1.0) -> dict[str, float]:
    """Une passe sur `loader`. Avec `optimizer` : entraînement ; sans : évaluation."""
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    with torch.set_grad_enabled(training):
        for seq, ctx, rul, state in loader:
            seq, ctx, rul, state = seq.to(device), ctx.to(device), rul.to(device), state.to(device)
            loss, mse, cross = combined_loss(model(seq, ctx), rul, state, lam, ce)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            totals += np.array([loss.item(), mse, cross]) * len(seq)
    loss, mse, cross = totals / len(loader.dataset)
    return {"loss": loss, "mse": mse, "ce": cross}


def fit(
    model, train_set, val_set, cfg: dict, device: str = "cpu", class_weights=None
) -> list[dict]:
    """Entraîne avec Adam, ReduceLROnPlateau et early stopping ; restaure les meilleurs poids."""
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.0)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    lam, clip = cfg.get("lam", 1.0), cfg.get("clip_norm", 1.0)
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(cfg["epochs"]):
        start = time.perf_counter()
        train = run_epoch(model, train_loader, device, lam, ce, optimizer, clip)
        seconds = time.perf_counter() - start
        val = run_epoch(model, val_loader, device, lam, ce)
        scheduler.step(val["loss"])
        history.append(
            {
                "epoch": epoch + 1,
                "seconds": seconds,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
        )
        if val["loss"] < best - 1e-5:
            best, best_state, wait = val["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def run_experiment(
    cfg: dict,
    seed: int,
    name: str | None = None,
    bundle: dict | None = None,
    results_dir: str = "results/runs",
    models_dir: str = "models",
) -> dict:
    """Entraîne une config avec une graine ; sauvegarde les poids et les scores de validation."""
    name = name or cfg["name"]
    # cfg["seed"] fixe le découpage train / validation ; `seed` initialise le modèle
    bundle = bundle or prepare(cfg["data"], cfg["seed"])
    rul_max = cfg["data"]["rul_max"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    model = build_model(
        cfg["model"], len(bundle["features"]), cfg["data"]["window"], bundle["scaler"].n_regimes
    ).to(device)
    counts = np.bincount(bundle["train"]["state"], minlength=3)
    class_weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    history = fit(
        model,
        CMAPSSDataset(bundle["train"], rul_max),
        CMAPSSDataset(bundle["val"], rul_max),
        cfg["training"],
        device,
        class_weights,
    )
    rul_pred, state_pred = predict(model, bundle["val"], rul_max, device)
    result = {
        "name": name,
        "seed": seed,
        "config": cfg,
        "n_params": count_parameters(model),
        "epochs": len(history),
        "sec_per_epoch": float(np.mean([h["seconds"] for h in history])),
        "val": regression_metrics(bundle["val"]["rul"], rul_pred),
        "history": history,
    }
    if state_pred is not None:
        result["val"].update(classification_metrics(bundle["val"]["state"], state_pred))
    save_checkpoint(model, Path(models_dir) / f"{name}_seed{seed}.pt")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / f"{name}_seed{seed}.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    return result


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Applique des surcharges du type `{"model.hidden_size": 32}` à une copie de la config."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        section, field = key.split(".")
        cfg[section][field] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne un modèle de RUL sur C-MAPSS")
    parser.add_argument("--config", default="configs/multitask.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    cfg = load_config(args.config)
    bundle = prepare(cfg["data"], cfg["seed"])
    scores = []
    for seed in args.seeds:
        result = run_experiment(cfg, seed, bundle=bundle)
        scores.append(result["val"]["rmse"])
        print(
            f"{result['name']} graine {seed} : RMSE val = {scores[-1]:.2f} "
            f"({result['epochs']} époques, {result['n_params']} paramètres)"
        )
    mean, std = np.mean(scores), np.std(scores)
    print(f"RMSE validation : {mean:.2f} ± {std:.2f} sur {len(scores)} graine(s)")


if __name__ == "__main__":
    main()
