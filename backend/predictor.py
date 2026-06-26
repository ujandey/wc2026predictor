"""Turn the trained artifacts into match predictions.

This is the webapp's equivalent of the notebook's ``predict_match`` helper:
given two teams it assembles the same feature vector used in training (current
ELO + each team's latest form snapshot) and returns calibrated W/D/L
probabilities from the home team's perspective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import HOME_ADVANTAGE, INITIAL_ELO, _empty_form


class Predictor:
    def __init__(self, artifacts: dict):
        self.model = artifacts["model"]
        self.le = artifacts["label_encoder"]
        self.feature_cols = artifacts["feature_cols"]
        self.elo = artifacts["elo_ratings"]
        self.form = artifacts["form_snapshot"]
        self.eligible_teams = artifacts["eligible_teams"]
        self.metrics = artifacts["metrics"]
        self.data_through = artifacts.get("data_through")
        # class index -> label ("W"/"D"/"L")
        self._classes = list(self.le.classes_)

    def get_elo(self, team: str) -> float:
        return self.elo.get(team, INITIAL_ELO)

    def _features(self, home: str, away: str, neutral: bool) -> pd.DataFrame:
        elo_home = self.get_elo(home)
        elo_away = self.get_elo(away)
        elo_home_adj = elo_home + (0 if neutral else HOME_ADVANTAGE)

        hf = self.form.get(home, _empty_form())
        af = self.form.get(away, _empty_form())

        row = {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home_adj - elo_away,
            "home_wins_last5": hf["wins_last5"],
            "home_losses_last5": hf["losses_last5"],
            "home_goals_scored_last5": hf["goals_scored_last5"],
            "home_goals_conceded_last5": hf["goals_conceded_last5"],
            "home_avg_goal_diff_last5": hf["avg_goal_diff_last5"],
            "home_clean_sheets_last5": hf["clean_sheets_last5"],
            "home_win_pct_last10": hf["win_pct_last10"],
            "away_wins_last5": af["wins_last5"],
            "away_losses_last5": af["losses_last5"],
            "away_goals_scored_last5": af["goals_scored_last5"],
            "away_goals_conceded_last5": af["goals_conceded_last5"],
            "away_avg_goal_diff_last5": af["avg_goal_diff_last5"],
            "away_clean_sheets_last5": af["clean_sheets_last5"],
            "away_win_pct_last10": af["win_pct_last10"],
            "is_friendly": 0,
        }
        return pd.DataFrame([row], columns=self.feature_cols).astype(float)

    def predict(self, home: str, away: str, neutral: bool = True) -> dict:
        """Return probabilities (home win / draw / away win) and context."""
        proba = self.model.predict_proba(self._features(home, away, neutral))[0]
        by_label = {self._classes[i]: float(p) for i, p in enumerate(proba)}
        p_home = by_label.get("W", 0.0)
        p_draw = by_label.get("D", 0.0)
        p_away = by_label.get("L", 0.0)
        return {
            "home_team": home,
            "away_team": away,
            "neutral": neutral,
            "prob_home_win": round(p_home, 4),
            "prob_draw": round(p_draw, 4),
            "prob_away_win": round(p_away, 4),
            "elo_home": round(self.get_elo(home), 1),
            "elo_away": round(self.get_elo(away), 1),
        }

    def probs_tuple(self, home: str, away: str, neutral: bool = True):
        """Lightweight (p_home, p_draw, p_away) for the simulator."""
        proba = self.model.predict_proba(self._features(home, away, neutral))[0]
        by_label = {self._classes[i]: float(p) for i, p in enumerate(proba)}
        return by_label.get("W", 0.0), by_label.get("D", 0.0), by_label.get("L", 0.0)

    def rankings(self, top: int | None = None) -> list[dict]:
        ranked = sorted(self.elo.items(), key=lambda kv: kv[1], reverse=True)
        if top:
            ranked = ranked[:top]
        return [
            {"rank": i, "team": team, "elo": round(elo, 1)}
            for i, (team, elo) in enumerate(ranked, 1)
        ]
