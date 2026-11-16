"""Chargement de C-MAPSS, cible RUL, découpage par moteur, normalisation, fenêtres."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
COLUMNS = ["unit", "cycle"] + SETTINGS + [f"s{i}" for i in range(1, 22)]
RUL_MAX = 125
# Nom physique des 14 capteurs utilisés (Saxena et al., 2008, tableau 2).
SENSOR_NAMES = {
    "s2": "T24 — température sortie compresseur BP",
    "s3": "T30 — température sortie compresseur HP",
    "s4": "T50 — température sortie turbine BP",
    "s7": "P30 — pression totale sortie compresseur HP",
    "s8": "Nf — vitesse de la soufflante",
    "s9": "Nc — vitesse du cœur",
    "s11": "Ps30 — pression statique sortie compresseur HP",
    "s12": "phi — débit carburant / Ps30",
    "s13": "NRf — vitesse corrigée de la soufflante",
    "s14": "NRc — vitesse corrigée du cœur",
    "s15": "BPR — taux de dilution",
    "s17": "htBleed — enthalpie de prélèvement",
    "s20": "W31 — débit de refroidissement turbine HP",
    "s21": "W32 — débit de refroidissement turbine BP",
}
# Classes d'état : 0 = sain (RUL > 80), 1 = à surveiller (30 < RUL <= 80), 2 = critique (<= 30).
STATE_THRESHOLDS = (30, 80)
STATE_NAMES = ["sain", "à surveiller", "critique"]


def load_cmapss(
    subset: str, split: str = "train", data_dir: str | Path = "data/raw"
) -> pd.DataFrame:
    """Charge `train_FDxxx.txt` ou `test_FDxxx.txt` avec des colonnes nommées."""
    path = Path(data_dir) / f"{split}_{subset}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier C-MAPSS introuvable : {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset: str, data_dir: str | Path = "data/raw") -> pd.Series:
    """Vraie RUL au dernier cycle observé de chaque moteur de test, indexée par `unit`."""
    values = np.loadtxt(Path(data_dir) / f"RUL_{subset}.txt")
    return pd.Series(values, index=np.arange(1, len(values) + 1), name="final_rul")


def rul_to_state(rul) -> np.ndarray:
    """Classe d'état dérivée de la RUL : 0 sain, 1 à surveiller, 2 critique."""
    rul = np.asarray(rul)
    low, high = STATE_THRESHOLDS
    return (2 - (rul > low).astype(int) - (rul > high).astype(int)).astype(np.int64)


def add_rul(
    df: pd.DataFrame, rul_max: int = RUL_MAX, final_rul: pd.Series | None = None
) -> pd.DataFrame:
    """Ajoute `rul_linear`, `rul` (plafonnée à `rul_max`) et `state`.

    Train : le dernier cycle est la panne. Test : la série s'arrête avant la panne,
    `final_rul` donne la RUL restante au dernier cycle observé.
    """
    out = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    remaining = df["unit"].map(final_rul) if final_rul is not None else 0
    out["rul_linear"] = last_cycle - df["cycle"] + remaining
    out["rul"] = out["rul_linear"].clip(upper=rul_max)
    out["state"] = rul_to_state(out["rul_linear"])
    return out


def split_by_unit(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42, fold: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare train / validation par identifiant de moteur (jamais par ligne).

    `fold` : numéro de pli d'une validation croisée à `1 / val_fraction` plis ; chaque moteur
    est alors en validation dans exactement un pli.
    """
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    if fold is None:
        val_units = rng.choice(units, size=max(1, round(len(units) * val_fraction)), replace=False)
    else:
        val_units = rng.permutation(units)[fold :: round(1 / val_fraction)]
    is_val = df["unit"].isin(val_units)
    return df[~is_val].copy(), df[is_val].copy()


@dataclass
class Scaler:
    """Standardisation ajustée sur le train, globale ou par régime de vol."""

    features: list[str]
    kmeans: KMeans | None  # None : une seule condition de vol
    per_regime: bool
    mean: np.ndarray  # (n_régimes ou 1, n_features)
    std: np.ndarray

    @property
    def n_regimes(self) -> int:
        return self.kmeans.n_clusters if self.kmeans is not None else 1

    def regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans is None:
            return np.zeros(len(df), dtype=np.int64)
        return self.kmeans.predict(df[SETTINGS].to_numpy()).astype(np.int64)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie une copie avec les capteurs normalisés et une colonne `regime`."""
        out = df.copy()
        regime = self.regimes(df)
        row = regime if self.per_regime else np.zeros_like(regime)
        out[self.features] = (df[self.features].to_numpy() - self.mean[row]) / self.std[row]
        out["regime"] = regime
        return out


def fit_scaler(
    train_df: pd.DataFrame, features: list[str], n_regimes: int = 1, per_regime: bool = True
) -> Scaler:
    """Ajuste la normalisation sur le train uniquement.

    `n_regimes > 1` : K-means sur les 3 réglages pour identifier les régimes de vol.
    `per_regime` : une moyenne / un écart-type par régime (sinon statistiques globales).
    """
    kmeans = None
    regime = np.zeros(len(train_df), dtype=np.int64)
    if n_regimes > 1:
        kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=0).fit(
            train_df[SETTINGS].to_numpy()
        )
        regime = kmeans.labels_
    groups = regime if per_regime else np.zeros_like(regime)
    values = train_df[features].to_numpy()
    n_groups = groups.max() + 1
    mean = np.stack([values[groups == g].mean(axis=0) for g in range(n_groups)])
    std = np.stack([values[groups == g].std(axis=0) for g in range(n_groups)])
    std[std < 1e-6] = 1.0  # capteur constant dans un régime : il devient 0 après centrage
    return Scaler(features, kmeans, per_regime and n_regimes > 1, mean, std)


def make_windows(
    df: pd.DataFrame, features: list[str], window: int, last_only: bool = False, n_regimes: int = 1
) -> dict[str, np.ndarray]:
    """Fenêtres glissantes de `window` cycles, moteur par moteur.

    La cible de chaque fenêtre est celle de son dernier cycle. `last_only` ne garde que la
    dernière fenêtre de chaque moteur (jeu de test). Un moteur plus court que `window` est
    complété au début en répétant sa première ligne : cela revient à supposer qu'il était
    dans le même état juste avant, sans inventer de dégradation.

    Renvoie X (n, window, n_features), ctx (n, n_regimes) en one-hot du régime du dernier
    cycle, rul, state, unit, cycle.
    """
    xs, tails = [], []
    for _, group in df.groupby("unit", sort=True):
        x = group[features].to_numpy(dtype=np.float32)
        if len(x) < window:
            x = np.concatenate([np.repeat(x[:1], window - len(x), axis=0), x])
        w = sliding_window_view(x, window, axis=0).transpose(0, 2, 1)
        tail = group.iloc[-len(w) :]  # ligne du dernier cycle de chaque fenêtre
        if last_only:
            w, tail = w[-1:], tail.iloc[-1:]
        xs.append(w)
        tails.append(tail)
    tail = pd.concat(tails)
    regime = tail["regime"].to_numpy() if "regime" in tail else np.zeros(len(tail), dtype=np.int64)
    return {
        "X": np.concatenate(xs).astype(np.float32),
        "ctx": np.eye(n_regimes, dtype=np.float32)[regime],
        "rul": np.array(tail["rul"], dtype=np.float32),  # np.array : copie modifiable pour torch
        "state": np.array(tail["state"], dtype=np.int64),
        "unit": tail["unit"].to_numpy(),
        "cycle": tail["cycle"].to_numpy(),
    }


class CMAPSSDataset(Dataset):
    """Renvoie `(sequence, contexte, rul normalisée, classe)` pour chaque fenêtre."""

    def __init__(self, windows: dict[str, np.ndarray], rul_max: int = RUL_MAX) -> None:
        self.seq = torch.from_numpy(windows["X"])
        self.ctx = torch.from_numpy(windows["ctx"])
        self.rul = torch.from_numpy(windows["rul"]) / rul_max  # cible d'ordre 1
        self.state = torch.from_numpy(windows["state"])

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.seq[index], self.ctx[index], self.rul[index], self.state[index]


def prepare(cfg: dict, seed: int = 42) -> dict:
    """Pipeline complet train / validation à partir de la section `data` d'une config."""
    df = add_rul(load_cmapss(cfg["subset"], "train", cfg["data_dir"]), cfg["rul_max"])
    train_df, val_df = split_by_unit(df, cfg["val_fraction"], seed, cfg.get("fold"))
    features = [f"s{i}" for i in cfg["sensors"]]
    scaler = fit_scaler(train_df, features, cfg["n_regimes"], cfg["per_regime"])
    return {
        "cfg": cfg,
        "features": features,
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
        "train": windows_for(train_df, scaler, cfg),
        "val": windows_for(val_df, scaler, cfg),
    }


def windows_for(
    df: pd.DataFrame, scaler: Scaler, cfg: dict, last_only: bool = False
) -> dict[str, np.ndarray]:
    """Normalise `df` avec un scaler déjà ajusté, puis fenêtre."""
    return make_windows(
        scaler.transform(df), scaler.features, cfg["window"], last_only, scaler.n_regimes
    )


def load_test(cfg: dict) -> pd.DataFrame:
    """Jeu de test officiel avec la RUL reconstruite à chaque cycle."""
    final_rul = load_test_rul(cfg["subset"], cfg["data_dir"])
    return add_rul(load_cmapss(cfg["subset"], "test", cfg["data_dir"]), cfg["rul_max"], final_rul)
