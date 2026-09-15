# progress_tracker.py
"""
Per-cell pipeline progress, so an interrupted run can resume from the
last fully completed step instead of starting over.

progress.json lives at outputs/<cell_name>/progress.json, one entry per
step with its status and any notes worth keeping (defaults used,
warnings, counts).
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _progress_path(outputs_dir: Path, cell_name: str) -> Path:
    return outputs_dir / cell_name / "progress.json"


def load_progress(outputs_dir: Path, cell_name: str) -> dict:
    path = _progress_path(outputs_dir, cell_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"cell_name": cell_name, "steps": {}}


def save_progress(outputs_dir: Path, cell_name: str, progress: dict):
    path = _progress_path(outputs_dir, cell_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def update_step(outputs_dir: Path, cell_name: str, step: str, status: str, notes: dict = None):
    """status: 'complete', 'error', 'skipped'."""
    progress = load_progress(outputs_dir, cell_name)
    progress["steps"][step] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes or {},
    }
    save_progress(outputs_dir, cell_name, progress)
    return progress


def is_step_complete(outputs_dir: Path, cell_name: str, step: str) -> bool:
    progress = load_progress(outputs_dir, cell_name)
    return progress.get("steps", {}).get(step, {}).get("status") == "complete"
