"""Build and cache the model artifacts.

Run directly (``python -m backend.train``) to regenerate. The API also calls
:func:`get_artifacts` lazily, building once on first launch if needed.
"""

from __future__ import annotations

import os
import pickle

from . import data, pipeline

ARTIFACT_PATH = os.path.join(data.DATA_DIR, "wc2026_model.pkl")


def build_artifacts(force: bool = False) -> dict:
    paths = data.ensure_data()
    print("[train] running pipeline (clean -> ELO -> form -> model) ...")
    artifacts = pipeline.build(paths["results.csv"])
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[train] saved -> {ARTIFACT_PATH}")
    print(f"[train] metrics: {artifacts['metrics']}")
    return artifacts


def get_artifacts() -> dict:
    """Load cached artifacts, building them on first use."""
    if os.path.exists(ARTIFACT_PATH):
        with open(ARTIFACT_PATH, "rb") as f:
            return pickle.load(f)
    return build_artifacts()


if __name__ == "__main__":
    build_artifacts(force=True)
