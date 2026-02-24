# IA-Hack2026-Groupe_8

## API FastAPI (meilleur modèle)

Cette API charge automatiquement le **meilleur modèle** depuis `models/model_validation_summary.csv` (trié par `f1_tuned`) puis sert des prédictions.

### 1) Installation

```bash
pip install -r requirements.txt
```

### 2) Lancer l'API

```bash
python -m uvicorn api.main:app --reload
```

### 3) Endpoints

- `GET /health` : statut API
- `GET /` : interface HTML de test
- `GET /model-info` : modèle chargé, seuil et features attendues
- `POST /predict` : prédictions batch

### 4) Interface web

Après lancement de l'API, ouvre :

- `http://127.0.0.1:8000/`

L'interface permet de :

- saisir les features du meilleur modèle,
- générer des données synthétiques en un clic,
- envoyer la requête de prédiction et visualiser la réponse JSON.

### 5) Exemple de requête

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"instances": [
			{
				"machine_type": "A",
				"maintenance_age_days": 120,
				"vib_mean": 0.58,
				"vib_std": 0.09,
				"vib_rms": 0.61,
				"temp_mean": 72.1,
				"temp_max": 86.4,
				"current_mean": 18.2,
				"current_peak": 24.5,
				"acoustic_energy": 135.7,
				"rpm_mean": 1490,
				"oil_particle_count": 210
			}
		]
	}'
```