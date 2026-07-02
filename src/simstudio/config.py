import os
from pathlib import Path
from datetime import datetime

def default_run_root() -> Path:
    root = os.environ.get("SIMSTUDIO_RUN_DIR")
    if root:
        return Path(root)
    tmp = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
    return tmp / "simstudio_runs"

def new_run_folder() -> Path:
    root = default_run_root()
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / ts
    run.mkdir(parents=True, exist_ok=True)
    return run
