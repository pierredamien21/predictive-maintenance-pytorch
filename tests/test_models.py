import numpy as np
import pytest
import torch

from src.data import CMAPSSDataset
from src.evaluate import bootstrap_rmse, nasa_score, regression_metrics
from src.models import GRURegressor, LSTMRegressor, MultiTaskRUL, build_model, count_parameters
from src.train import fit

SEQ, CTX = torch.randn(8, 30, 14), torch.eye(6)[torch.arange(8) % 6]


def test_nasa_score_worked_example():
    # d = -13 (prudent) et d = +10 (optimiste) coûtent chacun e - 1
    assert nasa_score([100, 100], [87, 110]) == pytest.approx(2 * (np.e - 1))
    assert nasa_score([50], [60]) > nasa_score([50], [40])  # l'optimisme est plus pénalisé


def test_rmse_and_mae():
    m = regression_metrics([0, 0], [3, -4])
    assert m["rmse"] == pytest.approx(np.sqrt(12.5)) and m["mae"] == 3.5


def test_parameter_counts_match_hand_computation():
    # LSTM : 4 portes * (14*64 + 64*64 + 2*64) = 20 480 ; GRU : 3 portes -> 15 360 ; tête : 65
    assert count_parameters(LSTMRegressor(14, hidden_size=64)) == 20_480 + 65
    assert count_parameters(GRURegressor(14, hidden_size=64)) == 15_360 + 65


@pytest.mark.parametrize(
    "cfg",
    [
        {"name": "mlp"},
        {"name": "lstm"},
        {"name": "gru", "num_layers": 2},
        {"name": "lstm", "bidirectional": True},
    ],
)
def test_regressors_return_one_value_per_window(cfg):
    assert build_model(cfg, 14, 30, 6)(SEQ, CTX).shape == (8,)


@pytest.mark.parametrize("use_context", [True, False])
def test_multitask_returns_rul_and_raw_logits(use_context):
    rul, logits = MultiTaskRUL(14, 6, use_context=use_context)(SEQ, CTX)
    assert rul.shape == (8,) and logits.shape == (8, 3)
    # logits bruts, pas des probabilités
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8))


def test_context_branch_changes_the_prediction():
    model = MultiTaskRUL(14, 6).eval()
    assert not torch.allclose(model(SEQ, CTX)[0], model(SEQ, CTX.roll(1, dims=0))[0])


def test_training_loop_overfits_a_single_batch():
    torch.manual_seed(0)
    windows = {
        "X": torch.randn(32, 30, 14).numpy(),
        "ctx": np.eye(6, dtype=np.float32)[np.arange(32) % 6],
        "rul": np.linspace(0, 125, 32, dtype=np.float32),
        "state": (np.arange(32) % 3).astype(np.int64),
    }
    data = CMAPSSDataset(windows)
    model = MultiTaskRUL(14, 6, dropout=0.0)
    cfg = {"batch_size": 32, "epochs": 200, "patience": 200, "learning_rate": 1e-2, "lam": 0.5}
    history = fit(model, data, data, cfg)
    assert history[-1]["val_loss"] < 0.05 * history[0]["val_loss"]


def test_bootstrap_interval_separates_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 125, 200)
    good, bad = y + rng.normal(0, 5, 200), y + rng.normal(0, 20, 200)
    rmse, low, high = bootstrap_rmse(y, good)
    assert low < rmse < high and rmse == pytest.approx(np.sqrt(np.mean((good - y) ** 2)))
    assert bootstrap_rmse(y, good, bad)[2] < 0  # « good » est meilleur : l'intervalle exclut 0
    assert bootstrap_rmse(y, good, good) == (0.0, 0.0, 0.0)
