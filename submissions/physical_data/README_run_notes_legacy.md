# SimStudio (Stage B) v0.3.0

Local desktop simulation editor (roads/lanes/vehicles/sensors) with live data + basic anomaly panel.

## Requirements

**Python 3.9 or newer.** Python 3.10+ recommended.

**Tkinter** must be included in your Python installation.
- macOS: use the official installer from https://www.python.org/downloads/ — it bundles tkinter.
  Homebrew Python (`brew install python`) and many Conda environments do **not** include tkinter.
  If you see `ModuleNotFoundError: No module named 'tkinter'`, switch to the python.org installer.
- Linux: install via `sudo apt install python3-tk` (Debian/Ubuntu) or `sudo dnf install python3-tkinter` (Fedora).
- Windows: tkinter is included with the standard python.org installer.

**Required packages** (installed automatically by `install.sh` or `pip install -e .`):
- `pillow>=9.0` — map tile rendering (Map Import window)
- `numpy>=1.21` — Kalman filter (required at startup)
- `openpyxl>=3.0` — Excel export (optional; app falls back to CSV if missing)

## Run (macOS / Linux)
```bash
# Install dependencies
pip3 install "pillow>=9.0" "numpy>=1.21" "openpyxl>=3.0"

# Run
python3 scripts/run_app.py
```

Or use the first-time setup script (macOS only):
```bash
bash install.sh          # installs packages, removes quarantine from .app bundle
```
Then double-click **Optical Fiber SIM.app**, or run `python3 scripts/run_app.py` directly.

## Run (Windows)
```bat
pip install "pillow>=9.0" "numpy>=1.21" "openpyxl>=3.0"
python scripts\run_app.py
```

## Key actions
- App opens empty. Use **New Scene** / **Load Scene** / **Create Demo**.
- Tools:
  - Select/Move: drag nodes (updates segments), drag vehicles (snaps to lane), drag sensors
  - Add Segment: click start/end; snaps to existing nodes to connect
  - Add Vehicle: click near a lane
  - Add GPS / Camera / DAS: click placement (camera: click+drag for direction; DAS: click segment)
- Delete selection: Delete/Backspace or Edit → Delete Selection
- Edit properties: select an object and use the right-side editor + Apply

## Recording
Nothing is written unless **Record run** is enabled. Output goes to a single temp run folder.

## Sensor-model notes in this package
- **GPS tab**: emits noisy world-position measurements `(x,y)` with range-dependent uncertainty `sigma_m`.
- **Camera tab**: emits noisy world-position estimates after a pinhole-style projection/back-projection approximation, with distance/FOV-dependent confidence.
- **DAS tab**: emits DAS-specific local features (distance to fiber, amplitude, SNR) and keeps the synthetic trace in the event payload for future DAS-map work.
