"""Data cleaning, ELO ratings, form features and model training.

This is a faithful, refactored port of the Kaggle notebook so the same logic
powers the live webapp. Running :func:`build` produces everything the API
needs: the trained (calibrated) classifier, current ELO ratings for every
national team, and a per-team "form snapshot" used to predict future matches.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config (mirrors the notebook)
# --------------------------------------------------------------------------
KEEP_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "Africa Cup of Nations",
    "Africa Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "CONCACAF Gold Cup",
    "Confederations Cup",
    "UEFA Nations League",
    "Friendly",
]

NAME_FIXES = {
    "Czech Republic": "Czechia",
    "China PR": "China",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
}

START_DATE = "1993-01-01"

INITIAL_ELO = 1500
K_FACTOR = 40
HOME_ADVANTAGE = 100

# Feature columns the model is trained on (order matters!).
FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "home_wins_last5", "home_losses_last5", "home_goals_scored_last5",
    "home_goals_conceded_last5", "home_avg_goal_diff_last5",
    "home_clean_sheets_last5", "home_win_pct_last10",
    "away_wins_last5", "away_losses_last5", "away_goals_scored_last5",
    "away_goals_conceded_last5", "away_avg_goal_diff_last5",
    "away_clean_sheets_last5", "away_win_pct_last10",
    "is_friendly",
]


# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
def load_and_clean(results_csv: str) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    results["date"] = pd.to_datetime(results["date"])

    results["is_friendly"] = results["tournament"] == "Friendly"
    results["is_worldcup"] = results["tournament"].str.contains(
        "FIFA World Cup", na=False
    )

    df = results[results["date"] >= START_DATE].copy()
    df = df[df["tournament"].isin(KEEP_TOURNAMENTS)].copy()

    # Remove bad rows
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])

    # Standardise team names
    df["home_team"] = df["home_team"].replace(NAME_FIXES)
    df["away_team"] = df["away_team"].replace(NAME_FIXES)

    # Target + helper columns
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        ["W", "D"],
        default="L",
    )
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["year"] = df["date"].dt.year

    df = df.sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2. ELO ratings
# --------------------------------------------------------------------------
def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def compute_elo(df: pd.DataFrame):
    """Run ELO chronologically. Returns (df_with_elo_features, elo_ratings)."""
    elo_ratings: dict[str, float] = {}
    history = []

    def get_elo(team: str) -> float:
        return elo_ratings.get(team, INITIAL_ELO)

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        is_neutral = bool(row.neutral)

        elo_home = get_elo(home)
        elo_away = get_elo(away)
        elo_home_adj = elo_home + (0 if is_neutral else HOME_ADVANTAGE)

        history.append({
            "date": row.date,
            "home_team": home,
            "away_team": away,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home_adj - elo_away,
        })

        if row.result == "W":
            score_home = 1.0
        elif row.result == "D":
            score_home = 0.5
        else:
            score_home = 0.0

        k = K_FACTOR * 1.5 if row.is_worldcup else K_FACTOR
        expected_home = expected_score(elo_home_adj, elo_away)
        change = k * (score_home - expected_home)

        bonus = 0 if is_neutral else HOME_ADVANTAGE
        elo_ratings[home] = (elo_home_adj + change) - bonus
        elo_ratings[away] = elo_away - change

    elo_df = pd.DataFrame(history)
    df = df.merge(elo_df, on=["date", "home_team", "away_team"], how="left")
    return df, elo_ratings


# --------------------------------------------------------------------------
# 3. Recent-form features (no future leakage)
# --------------------------------------------------------------------------
def _empty_form() -> dict:
    return {
        "wins_last5": 0, "losses_last5": 0, "draws_last5": 0,
        "goals_scored_last5": 0, "goals_conceded_last5": 0,
        "avg_goal_diff_last5": 0.0, "clean_sheets_last5": 0,
        "win_pct_last10": 0.0, "matches_played": 0,
    }


def _form_from_history(history: list[dict], n_short: int = 5, n_long: int = 10) -> dict:
    if not history:
        return _empty_form()
    last5 = history[-n_short:]
    last10 = history[-n_long:]
    return {
        "wins_last5": sum(1 for m in last5 if m["result"] == "W"),
        "losses_last5": sum(1 for m in last5 if m["result"] == "L"),
        "draws_last5": sum(1 for m in last5 if m["result"] == "D"),
        "goals_scored_last5": sum(m["goals_scored"] for m in last5),
        "goals_conceded_last5": sum(m["goals_conceded"] for m in last5),
        "avg_goal_diff_last5": round(sum(m["goal_diff"] for m in last5) / len(last5), 3),
        "clean_sheets_last5": sum(1 for m in last5 if m["goals_conceded"] == 0),
        "win_pct_last10": round(sum(1 for m in last10 if m["result"] == "W") / len(last10), 3),
        "matches_played": len(history),
    }


def compute_form(df: pd.DataFrame):
    """Add rolling form features. Returns (df, team_history) for later snapshots."""
    team_history: dict[str, list] = defaultdict(list)
    records = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        home_form = _form_from_history(team_history[home])
        away_form = _form_from_history(team_history[away])

        rec = {"date": row.date, "home_team": home, "away_team": away}
        rec.update({f"home_{k}": v for k, v in home_form.items()})
        rec.update({f"away_{k}": v for k, v in away_form.items()})
        records.append(rec)

        home_result = row.result
        away_result = "W" if home_result == "L" else ("L" if home_result == "W" else "D")
        team_history[home].append({
            "result": home_result,
            "goals_scored": row.home_score,
            "goals_conceded": row.away_score,
            "goal_diff": row.home_score - row.away_score,
        })
        team_history[away].append({
            "result": away_result,
            "goals_scored": row.away_score,
            "goals_conceded": row.home_score,
            "goal_diff": row.away_score - row.home_score,
        })

    form_df = pd.DataFrame(records)
    df = df.merge(form_df, on=["date", "home_team", "away_team"], how="left")
    return df, team_history


def latest_form_snapshot(team_history: dict[str, list]) -> dict[str, dict]:
    """Most recent form for every team — used to predict *future* matches."""
    return {team: _form_from_history(hist) for team, hist in team_history.items()}


# --------------------------------------------------------------------------
# 4. Model training (tuned + balanced + calibrated XGBoost)
# --------------------------------------------------------------------------
def train_model(df: pd.DataFrame, split_date: str = "2020-01-01"):
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    df_model = df[(df["home_matches_played"] >= 5) & (df["away_matches_played"] >= 5)].copy()

    X = df_model[FEATURE_COLS].astype(float)
    y = df_model["result"]

    train_mask = df_model["date"] < split_date
    test_mask = df_model["date"] >= split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    xgb_tuned = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_tuned.fit(X_train, y_train_enc, sample_weight=sample_weights, verbose=False)

    preds = le.inverse_transform(xgb_tuned.predict(X_test))
    acc = accuracy_score(y_test, preds)

    # Calibrate probabilities on the held-out period (notebook used isotonic).
    # sklearn >= 1.6 replaced cv="prefit" with FrozenEstimator; support both.
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(xgb_tuned), method="isotonic"
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(
            xgb_tuned, cv="prefit", method="isotonic"
        )
    calibrated.fit(X_test, y_test_enc)

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_date": split_date,
        "classes": list(le.classes_),
    }
    return calibrated, le, metrics


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def build(results_csv: str) -> dict:
    """Run the full pipeline and return an artifact dict ready to pickle."""
    df = load_and_clean(results_csv)
    df, elo_ratings = compute_elo(df)
    df, team_history = compute_form(df)
    model, label_encoder, metrics = train_model(df)
    form_snapshot = latest_form_snapshot(team_history)

    # Teams we trust enough to surface (>= 5 historical matches).
    eligible_teams = sorted(
        t for t, h in team_history.items() if len(h) >= 5
    )

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": FEATURE_COLS,
        "elo_ratings": elo_ratings,
        "form_snapshot": form_snapshot,
        "eligible_teams": eligible_teams,
        "metrics": metrics,
        "data_through": str(df["date"].max().date()),
    }
