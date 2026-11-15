# Predictive Maintenance with PyTorch

Projet de maintenance prédictive pour l'estimation de la durée de vie restante
(RUL) et la détection d'états de défaillance à partir de données moteur.

## Structure

- `data/` : données brutes et prétraitées, non versionnées
- `notebooks/` : exploration et expériences
- `src/` : chargement des données, modèles, entraînement et évaluation
- `configs/` : hyperparamètres YAML
- `models/` : poids entraînés, non versionnés
- `app/` : démonstration Streamlit

## Installation

```bash
python -m venv .venv
```

Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

```bash
python -m src.train --config configs/default.yaml
streamlit run app/streamlit_app.py
```

Les données doivent être placées dans `data/raw/`.
