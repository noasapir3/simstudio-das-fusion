"""Quick driver: simulate one of the test scenarios and emit the
new tracking-audit reports (DOCX + PDF) to `outputs/_sample_audit_report/`.

Run from the project root:

    PYTHONPATH=src python outputs/_run_sample_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simstudio.audit import export_all                     # noqa: E402
from simstudio.bus import EventBus                          # noqa: E402
from simstudio.project_io import load_world                 # noqa: E402
from simstudio.sim_core import Simulation                   # noqa: E402


SCENARIO = ROOT / "outputs" / "tracking" / "2seg_1car.sim.json"
OUT_DIR = ROOT / "outputs" / "_sample_audit_report"
SECONDS = 12.0
DT = 0.05


def main() -> None:
    if not SCENARIO.exists():
        raise SystemExit(f"scenario not found: {SCENARIO}")
    world = load_world(SCENARIO)
    bus = EventBus()
    captured: list = []
    for topic in (
        "world.vehicle_state", "world.vehicle_end",
        "sensor.gps", "sensor.camera", "sensor.das",
    ):
        bus.subscribe(topic, lambda ev, _topic=topic: captured.append(ev))
    sim = Simulation(bus, world)
    sim.rebuild_lanes()

    n = int(SECONDS / DT)
    for _ in range(n):
        sim.step(DT)

    print(f"captured {len(captured)} events from {SECONDS}s of simulation")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = export_all(captured, OUT_DIR, world=world)
    print("export_all result keys:", list(result.keys()))
    for k in ("audit_csv", "coverage_csv", "report_docx", "report_pdf"):
        v = result.get(k)
        if v:
            print(f"  {k}: {Path(v).name}  ({Path(v).stat().st_size:,} bytes)")
        else:
            print(f"  {k}: (not produced)")


if __name__ == "__main__":
    main()
