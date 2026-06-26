"""Monte-Carlo World Cup 2026 simulator.

Faithful to the notebook's fast simulator: match probabilities are pre-computed
once for every possible pairing, then thousands of tournaments are simulated by
sampling outcomes from those probabilities. Structure: 12 groups of 4 ->
group winners + runners-up + 8 best third-placed teams -> 32-team knockout.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


class Simulator:
    def __init__(self, predictor, groups: dict[str, list[str]]):
        self.predictor = predictor
        self.groups = groups
        self.teams = [t for ts in groups.values() for t in ts]
        self.match_probs = self._precompute()

    def _precompute(self) -> dict[tuple[str, str], tuple[float, float, float]]:
        """All group-stage matchups are at neutral venues (single host region)."""
        probs: dict[tuple[str, str], tuple[float, float, float]] = {}
        for a, b in combinations(self.teams, 2):
            ph, pdr, pa = self.predictor.probs_tuple(a, b, neutral=True)
            probs[(a, b)] = (ph, pdr, pa)
            # reverse orientation: home win <-> away win swap
            probs[(b, a)] = (pa, pdr, ph)
        return probs

    def _group(self, teams: list[str], rng: np.random.Generator):
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        for home, away in combinations(teams, 2):
            ph, pdr, pa = self.match_probs[(home, away)]
            r = rng.choice(3, p=[ph, pdr, pa])  # 0=home win, 1=draw, 2=away win
            if r == 0:
                pts[home] += 3; gd[home] += 1; gd[away] -= 1
            elif r == 1:
                pts[home] += 1; pts[away] += 1
            else:
                pts[away] += 3; gd[away] += 1; gd[home] -= 1
        ranked = sorted(
            teams,
            key=lambda t: (pts[t], gd[t], self.predictor.get_elo(t)),
            reverse=True,
        )
        return ranked, pts, gd

    def _knockout(self, a: str, b: str, rng: np.random.Generator) -> str:
        ph, pdr, pa = self.match_probs[(a, b)]
        # Redistribute the draw probability proportionally (penalties / ET).
        denom = ph + pa if (ph + pa) > 0 else 1.0
        ph_ko = ph + pdr * ph / denom
        return a if rng.random() < ph_ko else b

    def _one_tournament(self, rng: np.random.Generator):
        qualifiers = []
        thirds = []
        for teams in self.groups.values():
            ranked, pts, gd = self._group(teams, rng)
            qualifiers.append(ranked[0])
            qualifiers.append(ranked[1])
            thirds.append((ranked[2], pts[ranked[2]], gd[ranked[2]]))

        thirds.sort(
            key=lambda x: (x[1], x[2], self.predictor.get_elo(x[0])), reverse=True
        )
        bracket = qualifiers + [t[0] for t in thirds[:8]]

        while len(bracket) > 1:
            bracket = [
                self._knockout(bracket[i], bracket[i + 1], rng)
                for i in range(0, len(bracket) - 1, 2)
            ]
        return bracket[0]

    def run(self, n_sims: int = 10_000, seed: int = 42) -> list[dict]:
        rng = np.random.default_rng(seed)
        wins: dict[str, int] = {}
        for _ in range(n_sims):
            champ = self._one_tournament(rng)
            wins[champ] = wins.get(champ, 0) + 1
        results = sorted(wins.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {
                "team": team,
                "titles": count,
                "probability": round(count / n_sims * 100, 2),
            }
            for team, count in results
        ]
