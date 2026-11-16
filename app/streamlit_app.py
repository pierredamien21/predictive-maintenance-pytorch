"""Démo : RUL et état prédits pour un moteur du jeu de test.

streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # les configs utilisent des chemins relatifs à la racine du projet
sys.path.insert(0, str(ROOT))

from src.data import SENSOR_NAMES, STATE_NAMES, load_test, rul_to_state, windows_for  # noqa: E402
from src.evaluate import load_run, predict  # noqa: E402

MODELS = {
    "FD004 — GRU multi-tâche (modèle final)": "multitask_w40_fd004",
    "FD001 — GRU": "gru_best_fd001",
}
STATE_ADVICE = {
    "sain": "Aucune action.",
    "à surveiller": "Planifier la maintenance.",
    "critique": "Intervention rapide.",
}

st.set_page_config(page_title="Maintenance prédictive", page_icon="⚙️", layout="wide")
st.title("Maintenance prédictive de moteurs d'avion")
st.caption(
    "Durée de vie restante (RUL) et état d'un turboréacteur, prédits à partir de ses capteurs. "
    "Données : jeu simulé C-MAPSS de la NASA, moteurs du jeu de test officiel."
)


@st.cache_resource
def load(name: str):
    model, bundle, result = load_run(name, seed=0)
    return model, bundle, result, load_test(bundle["cfg"])


available = {
    label: name for label, name in MODELS.items() if Path(f"models/{name}_seed0.pt").exists()
}
if not available:
    st.error(
        "Aucun modèle entraîné. Lance : `python -m src.train --config configs/final_fd004.yaml`"
    )
    st.stop()

label = st.sidebar.selectbox("Modèle", list(available))
model, bundle, result, test_df = load(available[label])
cfg = bundle["cfg"]
unit = st.sidebar.selectbox("Moteur de test", sorted(test_df["unit"].unique()))
st.sidebar.divider()
st.sidebar.markdown(
    f"**Fenêtre** : {cfg['window']} cycles  \n**Paramètres** : {result['n_params']:,}".replace(
        ",", " "
    )
    + "  \n"
    f"**RMSE de validation** : {result['val']['rmse']:.1f} cycles"
)

engine = test_df[test_df["unit"] == unit]
windows = windows_for(engine, bundle["scaler"], cfg)
rul_pred, state_pred = predict(model, windows, cfg["rul_max"])
if state_pred is None:  # modèle mono-sortie : l'état est déduit de la RUL prédite
    state_pred = rul_to_state(rul_pred)
state = STATE_NAMES[state_pred[-1]]
error = rul_pred[-1] - windows["rul"][-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("RUL prédite", f"{rul_pred[-1]:.0f} cycles")
col2.metric("RUL réelle (plafonnée à 125)", f"{windows['rul'][-1]:.0f} cycles")
col3.metric(
    "Écart",
    f"{error:+.0f} cycles",
    "trop optimiste" if error > 0 else "trop pessimiste",
    delta_color="off",
)
col4.metric("État prédit", state.capitalize(), STATE_ADVICE[state], delta_color="off")

st.subheader("RUL au fil des cycles")
if len(engine) <= cfg["window"]:
    st.info(
        f"Ce moteur n'a que {len(engine)} cycles observés pour une fenêtre de {cfg['window']} : "
        "il n'y a qu'une seule prédiction, donc pas de courbe. La fenêtre est complétée en "
        "répétant le premier cycle."
    )
else:
    st.line_chart(
        pd.DataFrame(
            {"RUL prédite": rul_pred, "RUL réelle (plafonnée)": windows["rul"]},
            index=windows["cycle"],
        ),
        x_label="Cycle de vol",
        y_label="RUL (cycles)",
    )

st.subheader("Capteurs")
chosen = st.multiselect(
    "Capteurs à afficher",
    bundle["features"],
    default=["s11", "s4", "s12", "s7"],
    format_func=lambda s: f"{s} · {SENSOR_NAMES[s]}",
)
normalized = st.toggle("Valeurs normalisées (telles que vues par le modèle)", value=True)
source = bundle["scaler"].transform(engine) if normalized else engine
if chosen and normalized:
    st.line_chart(source.set_index("cycle")[chosen], x_label="Cycle de vol")
elif chosen:  # valeurs brutes : échelles très différentes, donc un graphique par capteur
    for column, sensor in zip(st.columns(len(chosen)), chosen, strict=True):
        column.caption(SENSOR_NAMES[sensor])
        column.line_chart(source.set_index("cycle")[sensor], x_label="Cycle de vol", height=220)
