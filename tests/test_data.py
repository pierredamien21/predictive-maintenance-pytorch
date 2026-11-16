import numpy as np
import pandas as pd
import pytest

from src.data import (
    COLUMNS,
    SETTINGS,
    add_rul,
    fit_scaler,
    load_cmapss,
    make_windows,
    rul_to_state,
    split_by_unit,
)

FEATURES = ["s2", "s3"]


def toy(lengths=(200, 50), n_regimes=1, seed=0):
    """Petit jeu synthétique : s2 vaut 1000 * unit + cycle, ce qui identifie chaque ligne."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit, n in enumerate(lengths, start=1):
        df = pd.DataFrame(0.0, index=range(n), columns=COLUMNS)
        df["unit"], df["cycle"] = unit, np.arange(1, n + 1)
        df["s2"] = 1000 * unit + df["cycle"]
        df["s3"] = rng.normal(size=n)
        df["setting_1"] = rng.integers(0, n_regimes, size=n) * 10.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_rul_is_capped_and_reaches_zero_at_failure():
    df = add_rul(toy(), rul_max=125)
    engine = df[df.unit == 1]
    assert engine.rul.iloc[0] == 125 and engine.rul_linear.iloc[0] == 199
    assert engine.rul.iloc[-1] == 0


def test_test_set_rul_adds_the_remaining_life():
    df = add_rul(toy(lengths=(40,)), final_rul=pd.Series({1: 20}))
    assert df.rul_linear.iloc[-1] == 20 and df.rul_linear.iloc[0] == 59


def test_state_thresholds():
    assert rul_to_state([81, 80, 31, 30, 0]).tolist() == [0, 1, 1, 2, 2]


def test_split_has_no_common_engine():
    df = toy(lengths=(40,) * 10)
    train, val = split_by_unit(df, 0.2, seed=1)
    assert set(train.unit).isdisjoint(val.unit)
    assert val.unit.nunique() == 2 and len(train) + len(val) == len(df)


def test_window_count_and_shape():
    w = make_windows(add_rul(toy(lengths=(200,))), FEATURES, window=30)
    assert w["X"].shape == (171, 30, 2)  # 200 - 30 + 1
    assert w["rul"][-1] == 0 and w["cycle"][0] == 30


def test_windows_never_span_two_engines():
    w = make_windows(add_rul(toy()), FEATURES, window=30)
    units_in_window = w["X"][:, :, 0] // 1000  # s2 encode le moteur
    assert (units_in_window == w["unit"][:, None]).all()
    # les cycles d'une fenêtre sont consécutifs et finissent au cycle cible
    assert (w["X"][:, -1, 0] % 1000 == w["cycle"]).all()
    assert (np.diff(w["X"][:, :, 0], axis=1) == 1).all()


def test_short_engine_is_padded_with_its_first_row():
    w = make_windows(add_rul(toy(lengths=(10,))), FEATURES, window=30, last_only=True)
    assert w["X"].shape == (1, 30, 2)
    assert (w["X"][0, :21, 0] == 1001).all() and w["X"][0, -1, 0] == 1010


def test_last_only_gives_one_window_per_engine():
    w = make_windows(add_rul(toy()), FEATURES, window=30, last_only=True)
    assert w["unit"].tolist() == [1, 2] and w["cycle"].tolist() == [200, 50]


def test_scaler_uses_train_statistics_only():
    train, other = toy(lengths=(100,)), toy(lengths=(100,), seed=1)
    other["s3"] += 50
    scaler = fit_scaler(train, FEATURES)
    assert np.allclose(scaler.transform(train)[FEATURES].mean(), 0, atol=1e-9)
    assert scaler.transform(other)["s3"].mean() > 10  # pas recentré sur ses propres données


def test_per_regime_scaling_removes_the_regime_offset():
    df = toy(lengths=(600,), n_regimes=3)
    df["s3"] += df["setting_1"] * 100  # le niveau du capteur dépend du régime
    scaler = fit_scaler(df, FEATURES, n_regimes=3)
    out = scaler.transform(df)
    assert out.regime.nunique() == 3
    assert np.allclose(out.groupby("regime")["s3"].mean(), 0, atol=1e-9)
    assert make_windows(add_rul(out), FEATURES, 30, n_regimes=3)["ctx"].shape[1] == 3


def test_real_fd004_has_six_clean_regimes():
    try:
        df = load_cmapss("FD004")
    except FileNotFoundError:
        pytest.skip("données C-MAPSS absentes")
    scaler = fit_scaler(df, ["s2"], n_regimes=6)
    spread = df[SETTINGS].groupby(scaler.regimes(df)).std().to_numpy()
    assert spread.max() < 0.01  # chaque groupe est un seul régime de vol


def test_cross_validation_folds_cover_each_engine_once():
    df = toy(lengths=(40,) * 10)
    val_units = [set(split_by_unit(df, 0.2, seed=1, fold=k)[1].unit) for k in range(5)]
    assert sorted(u for units in val_units for u in units) == list(range(1, 11))
