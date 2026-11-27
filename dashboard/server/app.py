"""Warpmate dashboard server: answers "analyze this position" for the page.

    uvicorn app:app --port 8000        (run from dashboard/server)

Environment variables (optional):
    WARPMATE_TABLES  directory of solved .wmtb tables (default: <repo>/tables)
    WARPMATE_MODEL   trained network (default: <repo>/model/warpmate_net.pt)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analysis import ROOT, Analyzer

TABLES = Path(os.environ.get("WARPMATE_TABLES", ROOT / "tables"))
MODEL = Path(os.environ.get("WARPMATE_MODEL", ROOT / "model" / "warpmate_net.pt"))

app = FastAPI(title="Warpmate dashboard")
analyzer = Analyzer(TABLES, MODEL)
rng = random.Random()


class AnalyzeRequest(BaseModel):
    fen: str


@app.get("/api/status")
def status() -> dict:
    """What's loaded: which tables, and whether the network is real."""
    return {
        "tables": sorted(analyzer.tables.tables),
        "network": "placeholder" if analyzer.network.placeholder else "trained",
        "model_path": str(MODEL),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    return analyzer.analyze(req.fen)


@app.get("/api/random")
def random_position(material: str | None = None) -> dict:
    try:
        return {"fen": analyzer.random_position(material, rng)}
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
