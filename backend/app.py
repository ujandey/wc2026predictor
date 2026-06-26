"""FastAPI app: serves the prediction/simulation API and the web frontend."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import groups as groups_mod
from .predictor import Predictor
from .simulator import Simulator
from .train import get_artifacts

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app = FastAPI(title="World Cup 2026 Predictor", version="1.0.0")

# Lazily-initialised globals (built on first request that needs them).
_predictor: Predictor | None = None
_groups: dict[str, list[str]] | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(get_artifacts())
    return _predictor


def get_groups() -> dict[str, list[str]]:
    global _groups
    if _groups is None:
        _groups = groups_mod.build_groups(get_predictor())
    return _groups


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    home: str
    away: str
    neutral: bool = True


class SimulateRequest(BaseModel):
    n_sims: int = Field(default=5000, ge=100, le=50000)
    seed: int = 42


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    p = get_predictor()
    return {
        "status": "ok",
        "data_through": p.data_through,
        "metrics": p.metrics,
        "n_teams": len(p.eligible_teams),
    }


@app.get("/api/teams")
def teams():
    return {"teams": get_predictor().eligible_teams}


@app.get("/api/rankings")
def rankings(top: int = 30):
    return {"rankings": get_predictor().rankings(top=top)}


@app.post("/api/predict")
def predict(req: PredictRequest):
    p = get_predictor()
    if req.home == req.away:
        raise HTTPException(400, "Pick two different teams.")
    for team in (req.home, req.away):
        if team not in p.elo:
            raise HTTPException(404, f"Unknown team: {team}")
    return p.predict(req.home, req.away, neutral=req.neutral)


@app.get("/api/groups")
def groups():
    return {"groups": get_groups()}


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    sim = Simulator(get_predictor(), get_groups())
    results = sim.run(n_sims=req.n_sims, seed=req.seed)
    return {"n_sims": req.n_sims, "results": results}


# --------------------------------------------------------------------------
# Frontend (static files)
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
