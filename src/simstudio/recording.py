import json
from pathlib import Path

class JsonlRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.fp = None

    def start(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.run_dir / "events.jsonl", "w", encoding="utf-8")

    def stop(self):
        if self.fp:
            self.fp.close()
            self.fp = None

    def write(self, ev):
        if not self.fp:
            return
        rec = {"topic": ev.topic, "ts": ev.ts, "payload": ev.payload}
        self.fp.write(json.dumps(rec) + "\n")
