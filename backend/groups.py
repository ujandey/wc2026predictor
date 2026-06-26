"""World Cup 2026 group definitions.

The 2026 finals expand to 48 teams across 12 groups of 4. The official draw
isn't hard-coded here (and the app shouldn't pretend to know a draw it can't
verify); instead groups are generated from the model itself — the 48
highest-rated eligible national teams, distributed across the 12 groups by
"snake" seeding so each group gets one team from each strength tier. A user can
override this with ``data/groups.json`` (a simple ``{"A": [...], ...}`` map).
"""

from __future__ import annotations

import json
import os
import string

from .data import DATA_DIR

GROUPS_OVERRIDE = os.path.join(DATA_DIR, "groups.json")

N_GROUPS = 12
TEAMS_PER_GROUP = 4
N_TEAMS = N_GROUPS * TEAMS_PER_GROUP  # 48
GROUP_NAMES = list(string.ascii_uppercase[:N_GROUPS])  # A..L

# Regional / non-FIFA sides that appear in the friendlies data but can't enter
# the World Cup — excluded from group seeding (still predictable head-to-head).
NON_FIFA = {
    "Basque Country", "Catalonia", "Galicia", "Andalusia",
    "Northern Cyprus", "Zanzibar", "Tibet", "Greenland",
    "Monaco", "Vatican City", "Western Sahara", "Kurdistan",
    "Provence", "Sápmi", "Padania", "Occitania",
}


def snake_seed(teams: list[str]) -> dict[str, list[str]]:
    """Distribute an ELO-ordered list of 48 teams into 12 groups, snake order."""
    groups: dict[str, list[str]] = {g: [] for g in GROUP_NAMES}
    order = GROUP_NAMES[:]
    for tier in range(TEAMS_PER_GROUP):
        tier_teams = teams[tier * N_GROUPS:(tier + 1) * N_GROUPS]
        seq = order if tier % 2 == 0 else order[::-1]
        for g, team in zip(seq, tier_teams):
            groups[g].append(team)
    return groups


def build_groups(predictor) -> dict[str, list[str]]:
    """Return WC2026 groups, honouring a user override if present."""
    if os.path.exists(GROUPS_OVERRIDE):
        with open(GROUPS_OVERRIDE) as f:
            return json.load(f)

    ranked = [r["team"] for r in predictor.rankings() if r["team"] not in NON_FIFA]
    top = ranked[:N_TEAMS]
    return snake_seed(top)
