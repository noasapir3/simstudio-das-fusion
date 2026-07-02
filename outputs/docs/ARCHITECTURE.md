# SimStudio Stage B — Architecture (v0.2.2)

- `models.py`: world graph (nodes/segments/lanes/vehicles/sensors)
- `sim_core.py`: stepping + synthetic sensor generation (GPS/Camera/DAS)
- `bus.py`: simple pub/sub
- `app.py`: editor-like GUI (toolbar, canvas, live/anomaly panels, properties editor)
- `scripts/run_app.py`: runs without needing installation (adds `src/` to sys.path)

Integration stubs: `integrations/fostmd_adapter.py`
