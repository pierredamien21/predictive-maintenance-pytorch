"""Registre des expériences du projet : comparaison FD001, grille, ablations FD004.

    python -m src.experiments --group fd001          # 5 graines par modèle
    python -m src.experiments --group grid --seeds 0 # recherche d'hyperparamètres
    python -m src.experiments --group fd004
    python -m src.experiments --summary              # tableau moyenne ± écart-type

Une expérience déjà terminée (fichier dans results/runs/) n'est pas relancée.
"""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

from .data import prepare
from .train import apply_overrides, run_experiment
from .utils import load_config

MT = "configs/multitask.yaml"
# nom -> (groupe, config de base, surcharges)
EXPERIMENTS = {
    "mlp_fd001": ("fd001", "configs/mlp.yaml", {}),
    "lstm_fd001": ("fd001", "configs/lstm.yaml", {}),
    "gru_fd001": ("fd001", "configs/gru.yaml", {}),
    "lstm2_fd001": ("fd001", "configs/lstm.yaml", {"model.num_layers": 2}),
    "bilstm_fd001": ("fd001", "configs/lstm.yaml", {"model.bidirectional": True}),
    # Configuration retenue par la grille (W = 40, hidden = 128), relancée avec 5 graines
    "lstm_best_fd001": (
        "fd001",
        "configs/lstm.yaml",
        {"data.window": 40, "model.hidden_size": 128},
    ),
    "gru_best_fd001": ("fd001", "configs/gru.yaml", {"data.window": 40, "model.hidden_size": 128}),
    "multitask_w40_fd004": ("fd004", "configs/final_fd004.yaml", {}),  # modèle final
    # FD004 : modèle final et ablations (une seule chose change à chaque fois)
    "multitask_fd004": ("fd004", MT, {}),
    "abl_monotask": ("fd004", MT, {"training.lam": 1.0}),  # régression seule
    "abl_no_context": ("fd004", MT, {"model.use_context": False}),
    "abl_lstm": ("fd004", MT, {"model.cell": "lstm"}),
    "abl_global_norm": ("fd004", MT, {"data.per_regime": False}),
    "lam_0.3": ("fd004", MT, {"training.lam": 0.3}),
    "lam_0.7": ("fd004", MT, {"training.lam": 0.7}),
}
# Grille limitée de la phase 5 (LSTM, FD001)
# Régression seule avec la fenêtre du modèle final : complète l'ablation multi-tâche
EXPERIMENTS["abl_monotask_w40"] = ("fd004", "configs/final_fd004.yaml", {"training.lam": 1.0})
# Validation croisée par moteur (5 plis) des deux modèles retenus : mesure le bruit du découpage
BEST_GRU = {"data.window": 40, "model.hidden_size": 128}
for fold in range(5):
    EXPERIMENTS[f"cv_gru_fd001_fold{fold}"] = (
        "cv",
        "configs/gru.yaml",
        BEST_GRU | {"data.fold": fold},
    )
    EXPERIMENTS[f"cv_multitask_fd004_fold{fold}"] = (
        "cv",
        "configs/final_fd004.yaml",
        {"data.fold": fold},
    )
for h, layers, w, d in itertools.product([32, 64, 128], [1, 2], [20, 30, 40], [0.1, 0.3]):
    EXPERIMENTS[f"grid_h{h}_l{layers}_w{w}_d{d}"] = (
        "grid",
        "configs/lstm.yaml",
        {"model.hidden_size": h, "model.num_layers": layers, "data.window": w, "model.dropout": d},
    )


def runs_table(results_dir: str = "results/runs") -> pd.DataFrame:
    """Une ligne par entraînement : scores de validation, nombre de paramètres, temps."""
    rows = []
    for path in Path(results_dir).glob("*_seed*.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": r["name"],
                "seed": r["seed"],
                "n_params": r["n_params"],
                "sec_per_epoch": r["sec_per_epoch"],
                **{k: v for k, v in r["val"].items() if k != "confusion"},
            }
        )
    return pd.DataFrame(rows)


def compare(
    name_a: str, name_b: str, metric: str = "rmse", runs: pd.DataFrame | None = None
) -> dict:
    """Écart `a - b` entre deux expériences et test de Welch sur leurs graines."""
    runs = runs_table() if runs is None else runs
    a = runs.loc[runs["name"] == name_a, metric].to_numpy()
    b = runs.loc[runs["name"] == name_b, metric].to_numpy()
    return {"diff": a.mean() - b.mean(), "p": float(ttest_ind(a, b, equal_var=False).pvalue)}


def summary(results_dir: str = "results/runs") -> pd.DataFrame:
    """Moyenne ± écart-type des scores de validation par expérience."""
    df = runs_table(results_dir)
    metrics = [c for c in df.columns if c not in ("name", "seed", "n_params")]
    out = df.groupby("name").agg(
        seeds=("seed", "count"),
        n_params=("n_params", "first"),
        **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")},
    )
    return out.sort_values("rmse_mean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance les expériences du projet")
    parser.add_argument("--group", choices=["fd001", "fd004", "grid", "cv"])
    parser.add_argument("--only", nargs="+", help="noms d'expériences précis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        print(summary().round(2).to_string())
        return
    names = args.only or [n for n, (group, _, _) in EXPERIMENTS.items() if group == args.group]
    for name in names:
        _, base, overrides = EXPERIMENTS[name]
        cfg = apply_overrides(load_config(base), overrides)
        bundle = None
        for seed in args.seeds:
            if Path(f"results/runs/{name}_seed{seed}.json").exists():
                continue
            bundle = bundle or prepare(cfg["data"], cfg["seed"])
            result = run_experiment(cfg, seed, name=name, bundle=bundle)
            print(f"{name} graine {seed} : RMSE val = {result['val']['rmse']:.2f}", flush=True)


if __name__ == "__main__":
    main()
