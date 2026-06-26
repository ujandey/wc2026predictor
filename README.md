# World Cup 2026 Predictor

A working webapp built from the original Kaggle notebook. It turns historical
international football results into:

- **ELO ratings** for every national team
- A **calibrated XGBoost** model that predicts any head-to-head match (win / draw / loss)
- A **Monte-Carlo simulator** that runs the 48-team World Cup 2026 thousands of
  times to estimate each team's title chances

The full notebook pipeline (cleaning → ELO → recent-form features → tuned,
class-balanced, calibrated XGBoost → tournament simulation) is reimplemented as
a FastAPI backend with a lightweight single-page frontend.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the model (downloads the dataset, runs the pipeline, caches artifacts)
python -m backend.train

# 3. Launch the app
uvicorn backend.app:app --reload

# open http://127.0.0.1:8000
```

Step 2 is optional — the API builds the model automatically on first launch if
the cached artifact (`data/wc2026_model.pkl`) is missing. Building takes a few
seconds and the result is cached, so subsequent launches are instant.

## How it works

| Stage | File | What it does |
|-------|------|--------------|
| Data | `backend/data.py` | Downloads `results.csv` (martj42 dataset, same source Kaggle re-publishes) and caches it under `data/raw/`. |
| Pipeline | `backend/pipeline.py` | Cleans data, computes ELO match-by-match (home advantage + World-Cup weighting), builds leak-free rolling form features, and trains the calibrated model. |
| Prediction | `backend/predictor.py` | Assembles the trained feature vector for any matchup and returns calibrated W/D/L probabilities. |
| Groups | `backend/groups.py` | Seeds the 48-team / 12-group bracket from the model's top teams (override with `data/groups.json`). |
| Simulator | `backend/simulator.py` | Pre-computes all matchup probabilities, then runs the group stage + 32-team knockout thousands of times. |
| API + UI | `backend/app.py`, `frontend/` | FastAPI endpoints and a vanilla-JS frontend (no build step). |

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Model metadata (data freshness, test accuracy). |
| `GET /api/teams` | All predictable teams. |
| `GET /api/rankings?top=30` | Current ELO rankings. |
| `POST /api/predict` | `{ "home": "Brazil", "away": "Germany", "neutral": true }` → W/D/L probabilities. |
| `GET /api/groups` | World Cup 2026 group draw used by the simulator. |
| `POST /api/simulate` | `{ "n_sims": 5000 }` → championship probability per team. |

## Customising the group draw

The official 48-team draw isn't hard-coded (the app won't claim a draw it can't
verify). By default groups are seeded from the 48 highest-rated national teams.
To use the real draw, create `data/groups.json`:

```json
{ "A": ["Mexico", "..."], "B": ["..."], "...": [] }
```

## Retraining

The model is fully reproducible from source data. To pull the latest results
and retrain:

```bash
python -m backend.train
```

## Notes

- Test accuracy is ~57% on a chronological 2020+ holdout — solid for three-way
  football outcomes where draws are inherently hard to call.
- All temporal features are computed strictly from past matches (no leakage).
