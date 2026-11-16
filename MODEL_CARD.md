# Fiche du modèle final

Modèle : `multitask_w40_fd004` (config `configs/final_fd004.yaml`, poids `models/multitask_w40_fd004_seed0.pt`).

## Ce que fait le modèle

- **Entrées** : les 40 derniers cycles de 14 capteurs d'un turboréacteur, normalisés par régime de vol, et le régime de vol du dernier cycle.
- **Sorties** : la durée de vie restante (RUL) en cycles, bornée à 125, et l'état du moteur (sain, à surveiller, critique).
- **Architecture** : GRU (taille cachée 64), petit MLP sur le régime de vol, concaténation, deux têtes linéaires. 15 796 paramètres.

## Usage prévu

Démonstration et étude de méthodes de pronostic sur le jeu **simulé** C-MAPSS FD004.

## Usages à exclure

- Toute décision de maintenance sur un moteur réel. Le modèle n'a jamais vu de données réelles.
- Les autres sous-ensembles (FD001, FD002, FD003) : la normalisation et les régimes sont ajustés sur FD004.
- La prédiction d'une RUL supérieure à 125 cycles : la cible d'entraînement est plafonnée.

## Données d'entraînement

- 199 moteurs de `train_FD004.txt` (80 % des moteurs) ; 50 moteurs en validation.
- Découpage par moteur ; normalisation et identification des régimes (K-means à 6 groupes) ajustées sur le train uniquement.

## Performances

Jeu de test officiel de FD004 (248 moteurs), RUL plafonnée à 125, moyenne sur 5 graines.

| Mesure | Valeur |
| --- | --- |
| RMSE | 14,08 cycles, intervalle de confiance à 95 % : [12,38 ; 15,87] |
| MAE | 10,36 cycles |
| Score NASA | 871 |
| F1 macro de la classification | 0,85 |
| Rappel de la classe critique | 0,94 |
| Écart de RMSE avec un Random Forest réglé | +0,07 [−0,84 ; +0,97] : non significatif |

Sur la validation : RMSE de 8,4 cycles quand la RUL réelle est inférieure ou égale à 30, de 16,0 entre 31 et 124.

## Limites connues

- **Biais optimiste** : +6,8 cycles en moyenne quand la RUL réelle est inférieure ou égale à 80. Le modèle annonce plus de marge qu'il n'en reste.
- **Moteurs loin de la panne** : les plus grosses erreurs sont des sous-estimations de moteurs dont la RUL réelle dépasse 95 cycles.
- **Pas d'incertitude** : le modèle donne une valeur, sans intervalle.
- **Alternative plus simple** : un Random Forest sur des caractéristiques de fenêtre atteint la même RMSE pour une fraction du temps de calcul. Le réseau n'apporte que la sortie de classification.
- **Sensibilité au découpage** : en validation croisée à 5 plis, la RMSE varie de 13,2 à 16,2 selon les moteurs tirés en validation.

Le détail des protocoles et des menaces à la validité est dans `notebooks/03_experiences.ipynb` et `notebooks/04_evaluation.ipynb`.
