# DAS map (space–time heatmap)

The paper visualizes DAS as a **space–time map**: distance along the fiber (channels) vs. time, with color = signal amplitude.

In SimStudio, you can export the DAS table and plot a similar heatmap externally (without changing the GUI).

## 1) Export
In the app: **Data → Export All…**

This creates a folder like:

- `simstudio_export_YYYY-MM-DD_HHMMSS/`
  - `das.csv`

## 2) Plot heatmap
From the project root (the folder that has `pyproject.toml`):

```bash
python3 scripts/plot_das_map.py --csv /path/to/simstudio_export_*/das.csv
```

Or automatically pick the newest export folder:

```bash
python3 scripts/plot_das_map.py --export_root /path/to/parent --latest
```

By default it plots `das_amplitude`. To plot SNR instead:

```bash
python3 scripts/plot_das_map.py --csv /path/to/das.csv --value snr
```

Output: a PNG next to the CSV (e.g., `das.png`).

### Notes
- **Y axis** is `sensor_id` (sorted). If later you add real *fiber position* for each sensor/channel, we can swap the y-axis to meters.
- Missing (sensor_id, t) samples are filled with 0.
