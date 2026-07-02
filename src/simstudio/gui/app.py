import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import math
import random
import json
import csv
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    _OPENPYXL_OK = True
except ImportError:  # pragma: no cover
    _OPENPYXL_OK = False

try:
    from simstudio.export_xlsx import export_workbook as _export_workbook
    _EXPORT_OK = True
except Exception:  # pragma: no cover
    _EXPORT_OK = False
from typing import Optional
from queue import Queue, Empty
from collections import deque
from pathlib import Path
import urllib.parse
import urllib.request
from io import BytesIO

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MPL_OK = True
except Exception:  # pragma: no cover
    Figure = None  # type: ignore
    FigureCanvasTkAgg = None  # type: ignore
    _MPL_OK = False
from ..bus import EventBus
from ..models import World, Node, Segment, Vehicle, GPSSensor, CameraSensor, DASSensor
from ..kalman import build_kalman_rows_tracked, build_kalman_rmse, build_sensor_gid_lookup
from .. import audit as _audit_mod
from ..sim_core import Simulation
from ..geometry import polyline_nearest_s, dist, point_at_s, fov_wedge, polyline_length
from ..project_io import save_world, load_world, world_to_dict, dict_to_world
from ..randomizer import randomize_world, RandomizeSpec, spawn_vehicles_over_time


TOOL_SELECT = "select"
TOOL_SEGMENT = "segment"
TOOL_VEHICLE = "vehicle"
TOOL_ROUTE_VEHICLE = "route_vehicle"  # legacy (kept for compatibility)
TOOL_GPS = "gps"
TOOL_CAMERA = "camera"
TOOL_DAS = "das"
TOOL_BOUNDARY_EXIT = "boundary_exit"  # toggle a node as a map-edge exit point

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _random_vehicle_macro_profile(speed_limit_mps: float = 13.9) -> dict:
    """Return a simple, realistic per-vehicle speed profile.

    The profile is intentionally UI-friendly: the user controls only min/avg/max
    speed, max acceleration / braking, and the interval between target-speed
    changes. Internally the simulator samples a new target speed every few
    seconds and drives toward it with bounded acceleration.
    """
    speed_limit_mps = max(6.0, float(speed_limit_mps or 13.9))
    speed_limit_kmh = speed_limit_mps * 3.6

    class_pick = random.random()
    if class_pick < 0.25:
        min_kmh = random.uniform(22.0, 38.0)
        avg_kmh = random.uniform(max(min_kmh + 6.0, 35.0), 52.0)
        max_kmh = random.uniform(max(avg_kmh + 8.0, 55.0), min(80.0, speed_limit_kmh * 1.02 + 10.0))
        interval_min = random.uniform(4.0, 7.0)
        interval_max = random.uniform(max(interval_min + 2.0, 8.0), 14.0)
        max_accel = random.uniform(1.0, 1.8)
        max_decel = random.uniform(1.8, 3.0)
    elif class_pick < 0.80:
        min_kmh = random.uniform(28.0, 45.0)
        avg_kmh = random.uniform(max(min_kmh + 8.0, 42.0), 68.0)
        max_kmh = random.uniform(max(avg_kmh + 10.0, 65.0), min(105.0, speed_limit_kmh * 1.08 + 12.0))
        interval_min = random.uniform(3.0, 6.0)
        interval_max = random.uniform(max(interval_min + 2.0, 7.0), 12.0)
        max_accel = random.uniform(1.4, 2.6)
        max_decel = random.uniform(2.3, 4.0)
    else:
        min_kmh = random.uniform(35.0, 55.0)
        avg_kmh = random.uniform(max(min_kmh + 8.0, 55.0), 82.0)
        max_kmh = random.uniform(max(avg_kmh + 10.0, 80.0), min(120.0, speed_limit_kmh * 1.12 + 15.0))
        interval_min = random.uniform(2.5, 5.0)
        interval_max = random.uniform(max(interval_min + 2.0, 6.0), 10.0)
        max_accel = random.uniform(1.8, 3.2)
        max_decel = random.uniform(2.8, 4.8)

    max_kmh = max(max_kmh, avg_kmh + 6.0, min_kmh + 12.0)
    avg_kmh = min(max(avg_kmh, min_kmh + 4.0), max_kmh - 4.0)

    speed_min = min_kmh / 3.6
    speed_mean = avg_kmh / 3.6
    speed_max = max_kmh / 3.6
    target_speed = _clamp(random.gauss(speed_mean, max(0.5, 0.22 * (speed_max - speed_min))), speed_min, speed_max)
    interval_mean = 0.5 * (interval_min + interval_max)
    speed_std = max(0.5, 0.18 * (speed_max - speed_min))

    return {
        "speed_min_mps": speed_min,
        "speed_max_mps": speed_max,
        "speed_mean_mps": speed_mean,
        "speed_std_mps": speed_std,
        "target_speed_mps": target_speed,
        "speed_change_interval_mean_s": interval_mean,
        "speed_change_interval_min_s": interval_min,
        "speed_change_interval_max_s": interval_max,
        "next_speed_change_t": 0.0,
        "last_speed_change_t": -1e9,
        "cruise_hold_probability": 0.0,
        "accel_response_s": 1.5,
        "max_accel_mps2": max_accel,
        "max_decel_mps2": max_decel,
    }



MAP_MIN_ZOOM = 17
MAP_MAX_SEGMENTS = 400

class Viewport:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0

    def world_to_screen(self, wx, wy, cw, ch):
        sx = (wx + self.ox) * self.scale + cw / 2
        sy = (wy + self.oy) * self.scale + ch / 2
        return sx, sy

    def screen_to_world(self, sx, sy, cw, ch):
        wx = (sx - cw / 2) / self.scale - self.ox
        wy = (sy - ch / 2) / self.scale - self.oy
        return wx, wy

    def fit_to_world(self, world, cw, ch, padding: float = 0.15):
        """Set ox, oy, scale so all world content is centred and fits the canvas.

        Collects the bounding box from nodes (which anchor every road segment)
        plus fixed sensors (GPS, cameras).  Falls back to reset() when the world
        is empty or the canvas has not yet been sized.
        """
        xs, ys = [], []
        for n in world.nodes.values():
            xs.append(float(n.x)); ys.append(float(n.y))
        for g in world.gps.values():
            xs.append(float(g.x)); ys.append(float(g.y))
        for c in world.cameras.values():
            xs.append(float(c.x)); ys.append(float(c.y))

        if not xs or cw <= 1 or ch <= 1:
            self.reset()
            return

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        self.ox = -cx
        self.oy = -cy

        world_w = max(xmax - xmin, 1.0)
        world_h = max(ymax - ymin, 1.0)
        scale_x = cw * (1.0 - padding) / world_w
        scale_y = ch * (1.0 - padding) / world_h
        self.scale = max(0.25, min(6.0, min(scale_x, scale_y)))


class UndoRedo:
    def __init__(self, max_depth=80):
        self.max_depth = max_depth
        self.undo = []
        self.redo = []
        self._last = None

    def snapshot(self, world: World):
        return json.dumps(world_to_dict(world))

    def push(self, world: World):
        snap = self.snapshot(world)
        if self._last == snap:
            return
        self.undo.append(snap)
        if len(self.undo) > self.max_depth:
            self.undo = self.undo[-self.max_depth:]
        self.redo = []
        self._last = snap

    def can_undo(self):
        return len(self.undo) > 0

    def can_redo(self):
        return len(self.redo) > 0


    def clear(self):
        self.undo = []
        self.redo = []
        self._last = None

    def do_undo(self, current_world: World):
        if not self.undo:
            return None
        cur = self.snapshot(current_world)
        self.redo.append(cur)
        snap = self.undo.pop()
        self._last = snap
        return dict_to_world(json.loads(snap))

    def do_redo(self, current_world: World):
        if not self.redo:
            return None
        cur = self.snapshot(current_world)
        self.undo.append(cur)
        snap = self.redo.pop()
        self._last = snap
        return dict_to_world(json.loads(snap))


class _FakeTv:
    """Frozen snapshot of a Treeview's data, safe to pass to a background thread.

    Supports the minimal subset of the ttk.Treeview API used by the export
    functions: ``tv["columns"]``, ``tv.get_children()``, and
    ``tv.item(iid, "values")``.  Instances are created on the main thread and
    then handed off to the worker; no Tkinter calls occur after construction.
    """

    def __init__(self, columns, rows):
        self._columns = tuple(columns)
        # rows is a list of lists; store as tuple-of-tuples for immutability.
        self._rows = [tuple(r) for r in rows]

    def __getitem__(self, key):
        if key == "columns":
            return self._columns
        raise KeyError(key)  # pragma: no cover

    def get_children(self):
        return range(len(self._rows))

    def item(self, iid, attr="values"):
        if attr == "values":
            return self._rows[iid]
        return {}  # pragma: no cover


def _snapshot_tv(tv) -> "_FakeTv":
    """Capture a Treeview's columns and rows into a thread-safe _FakeTv.

    Must be called on the main (Tkinter) thread.
    """
    cols = list(tv["columns"])
    rows = [list(tv.item(iid, "values")) for iid in tv.get_children()]
    return _FakeTv(cols, rows)


class SimStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SimStudio — Fiber Optic Traffic Lab")
        self.geometry("1500x900")
        self.minsize(1200, 720)

        # Set application icon from the project logo.
        try:
            _icon_path = Path(__file__).parent.parent / "assets" / "icon_256.png"
            if _icon_path.exists():
                _icon_img = tk.PhotoImage(file=str(_icon_path))
                self.iconphoto(True, _icon_img)
                self._icon_img = _icon_img  # keep reference so GC doesn't collect it
        except Exception:
            pass  # icon is optional — never crash the app over it

        self.bus = EventBus()
        self.world = World()
        self.sim = Simulation(self.bus, self.world)
        self._scene_name: str = "Untitled"  # current simulation/scene name
        self._scene_json_path = None        # Path to the loaded .sim.json (for anomaly features)

        self.ev_q = Queue()
        self._all_events = []          # raw list of Event objects
        # Persistent anomaly marker state — populated in _ui_pump as events
        # arrive, cleared on reset.  These outlive the _all_events rolling
        # window so markers never vanish mid-simulation.
        self._persistent_collided_ids: set = set()       # vehicle IDs in any collision
        self._persistent_route_complete_ids: set = set() # vehicle IDs that finished route
        self._persistent_stuck_positions: list = []      # list of (x, y) world-coords
        self._snap_lock = threading.Lock()
        self._snaps = []  # list of sim snapshots (for timeline scrubbing)
        self._playhead = -1
        self._playback_mode = False
        self._dirty_from_scrub = False
        # Route planning bookkeeping (UI-only)
        self._vehicle_routes = {}   # vid -> list of (lane_id, s)
        self._route_progress = {}  # vid -> int (how many waypoints were reached)
        self._data_ready = False
        try:
            self._open_end_nodes.clear()
        except Exception:
            pass
        self._ever_played = False
        self._open_end_nodes = set()

        # Timeline / scrubbing (replay after Stop)
        self.view_time = None
        self.time_var = tk.DoubleVar(value=0.0)
        self._time_max = 0.0
        self._scrubbing = False

        for topic in [
            "world.vehicle_state",
            "world.vehicle_stuck",
            "world.route_waypoint_reached",
            "world.route_complete",   # anomaly: clean route end (blue X marker).
            "world.collision",        # anomaly: collision pair (red warning triangle).
            "sensor.gps",
            "sensor.camera",
            "sensor.das",
        ]:
            self.bus.subscribe(topic, lambda ev, t=topic: self.ev_q.put(ev))

        # -------------------------------------------------------------
        # Phase 1: invisible cross-segment vehicle-tracking subscriber.
        #
        # The TrackManager listens on the same bus topics and builds a
        # parallel view of vehicle identity (global_track_id).  In Phase 1
        # it does NOT publish events, does NOT mutate payloads, and is NOT
        # consulted by the GUI, Kalman pipeline, or exports — it exists
        # only to accumulate state that later phases will surface.
        #
        # The entire block is wrapped defensively: any failure here must
        # not affect the rest of the application.
        # -------------------------------------------------------------
        self._track_manager = None
        try:
            from ..tracking import TrackManager as _TrackManager
            # Phase 2a: give the manager a bus reference so it can publish
            # ``track.birth`` / ``track.update`` on the new ``track.*``
            # topic namespace.  The GUI does not yet subscribe to those
            # topics, so this remains invisible to the user.
            self._track_manager = _TrackManager(self.world, bus=self.bus)
        except Exception:
            self._track_manager = None

        if self._track_manager is not None:
            _tm = self._track_manager
            for _topic in (
                "world.vehicle_state",
                "world.vehicle_stuck",
                "sensor.gps",
                "sensor.camera",
                "sensor.das",
            ):
                def _tm_cb(ev, _m=_tm):
                    try:
                        _m.on_event(ev)
                    except Exception:
                        # Tracker must never propagate errors to the bus.
                        pass
                self.bus.subscribe(_topic, _tm_cb)

        # Phase 2b: separate bounded ring buffer for track.* events.  We
        # intentionally do NOT route these through ev_q because their rate
        # (roughly the sum of all sensor rates × n_vehicles) can spike and
        # would otherwise delay the primary event drain.  The ring buffer
        # drops the oldest element when full — a bounded memory guarantee.
        # The maxlen is conservative: at 1 kHz emission it holds ~20 s of
        # history before any drops occur.
        self._track_buffer: "deque" = deque(maxlen=20000)
        try:
            def _track_sink(ev, _buf=self._track_buffer):
                try:
                    _buf.append(ev)
                except Exception:
                    pass
            self.bus.subscribe("track.birth", _track_sink)
            self.bus.subscribe("track.update", _track_sink)
        except Exception:
            pass

        self.tool = tk.StringVar(value=TOOL_SELECT)
        # Segment creation params
        self.segment_lanes = tk.IntVar(value=1)  # 1..7
        self.vehicle_add_mode = tk.StringVar(value="free")  # free | routed
        self.status = tk.StringVar(value="File → New Scene / Load Scene. (Create Demo is a quick example map)")
        self.selected = None
        self.pending = None

        # Traffic annotation (segment tagging) UI mode
        self._traffic_mark_active = False  # only true while Random Populate dialog is tagging
        self._traffic_mark_level = None  # None|"light"|"medium"|"heavy"
        self._traffic_last_seg = None
        self._traffic_last_ts = 0.0

        self.dragging_node = None
        self.dragging_vehicle = None
        self.dragging_sensor = None
        self.dragging_pan = None

        self.vp = Viewport()
        self._running = False
        self._review_mode = False  # timeline scrubbed to past: view-only
        self.view_time = float(self.time_var.get())
        self._last_tick = None

        self.undo = UndoRedo()
        self._drag_snapshot_taken = False
        self._drag_world_before = None

        self._build_ui()
        self._build_menu()
        self._bind_shortcuts()

        # Vehicle spawning over time (configured via Random Populate dialog)
        self._rand_spawn_spec = None
        self._rand_spawn_state = {"acc": 0.0}
        # Render throttling: coalesce multiple "please redraw" requests into a
        # single frame. This prevents Tk's event loop from getting starved and
        # causing "clicks apply later" behavior under heavy event rates.
        self._render_pending = False
        self._render()
        self.after(30, self._ui_pump)

    def _request_render(self):
        if self._render_pending:
            return
        self._render_pending = True
        self.after_idle(self._render_frame)

    def _render_frame(self):
        self._render_pending = False
        self._render()

    def _update_window_title(self):
        """Sync the OS window title to the current scene name."""
        self.title(f"SimStudio — {self._scene_name}")

    # ---------------- UI ----------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.nb = ttk.Notebook(self)
        self.nb.grid(row=0, column=0, sticky="nsew")

        self.tab_sim = ttk.Frame(self.nb)
        self.tab_data = ttk.Frame(self.nb)
        self.nb.add(self.tab_sim, text="Simulation")
        self.nb.add(self.tab_data, text="Data")

        # Simulation tab layout
        self.tab_sim.columnconfigure(0, weight=1)
        self.tab_sim.rowconfigure(0, weight=1)

        pan = ttk.Panedwindow(self.tab_sim, orient=tk.HORIZONTAL)
        pan.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.left = ttk.Frame(pan, width=230)
        self.center = ttk.Frame(pan)
        self.right = ttk.Frame(pan, width=280)

        pan.add(self.left, weight=0)
        pan.add(self.center, weight=5)
        pan.add(self.right, weight=1)

        self._build_left()
        self._build_center()
        self._build_right()

        # Data tab layout
        self._build_data_tab()


    def _build_left(self):
        self.left.columnconfigure(0, weight=1)
        ttk.Label(self.left, text="Tools", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self._tool_btn("Select / Move (V)", TOOL_SELECT, 1)
        self._tool_btn("Add Segment (S)", TOOL_SEGMENT, 2)
        self._tool_btn("Add Vehicle (R)", TOOL_VEHICLE, 3)
        self.seg_menu = ttk.LabelFrame(self.left, text="Road mode")
        self.seg_menu.grid(row=4, column=0, sticky="ew", padx=10, pady=(10, 6))
        self.seg_menu.columnconfigure(0, weight=1)
        ttk.Label(self.seg_menu, text="All roads are single-lane and one-way.", wraplength=180).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 6))

        self._tool_btn("Add GPS (G)", TOOL_GPS, 6)
        self._tool_btn("Add Camera (C)", TOOL_CAMERA, 7)
        self._tool_btn("Add DAS (D)", TOOL_DAS, 8)
        self._tool_btn("Exit Node (X)", TOOL_BOUNDARY_EXIT, 9)

        ttk.Separator(self.left).grid(row=10, column=0, sticky="ew", padx=10, pady=10)
        ttk.Label(self.left, text="Run", font=("Helvetica", 12, "bold")).grid(row=11, column=0, sticky="w", padx=10, pady=(0, 6))
        ttk.Button(self.left, text="Start", command=self._start).grid(row=12, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Stop", command=self._stop).grid(row=13, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Step (20ms)", command=self._step_once).grid(row=14, column=0, sticky="ew", padx=10, pady=4)

        ttk.Separator(self.left).grid(row=15, column=0, sticky="ew", padx=10, pady=10)
        ttk.Button(self.left, text="New Scene", command=self._new_scene).grid(row=16, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Create Demo", command=self._create_demo_scene).grid(row=17, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Random Populate…", command=self._random_populate).grid(row=18, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Save Scene...", command=self._save_scene).grid(row=19, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Load Scene...", command=self._load_scene).grid(row=20, column=0, sticky="ew", padx=10, pady=4)
        ttk.Button(self.left, text="Import Roads… (M)", command=self._import_roads_from_map).grid(row=21, column=0, sticky="ew", padx=10, pady=4)

        ttk.Label(self.left, textvariable=self.status, wraplength=210).grid(row=22, column=0, sticky="ew", padx=10, pady=(12, 10))

    def _tool_btn(self, label, value, row):
        rb = ttk.Radiobutton(self.left, text=label, value=value, variable=self.tool, command=self._tool_changed)
        rb.grid(row=row, column=0, sticky="w", padx=10, pady=2)

    def _build_center(self):
        self.center.columnconfigure(0, weight=1)
        self.center.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.center, bg="#0f1115", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        self.canvas.bind("<Button-2>", self._on_pan_down)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_up)
        self.canvas.bind("<Button-3>", self._on_pan_down)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_up)

        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(0.9, e.x, e.y))

        # Re-render whenever the canvas gets its real size (first map) or is resized.
        # This fixes the startup bug where _render() in __init__ runs before the
        # geometry manager has assigned real dimensions (winfo_width returns 1 at
        # that point, and `1 or 900` evaluates to 1, not 900).
        self.canvas.bind("<Configure>", lambda e: self._request_render())

        # Timeline bar (under canvas)
        bar = ttk.Frame(self.center)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="t").grid(row=0, column=0, sticky="w")
        self._time_label = ttk.Label(bar, text="0.00s")
        self._time_label.grid(row=0, column=2, sticky="e", padx=(8,0))
        self.time_scale = ttk.Scale(bar, from_=0.0, to=0.0, variable=self.time_var, command=self._on_time_scrub)
        self.time_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.time_scale.bind("<ButtonPress-1>", lambda e: setattr(self, "_scrubbing", True))
        self.time_scale.bind("<ButtonRelease-1>", self._on_time_scrub_end)

    def _build_right(self):
        self.right.columnconfigure(0, weight=1)
        ttk.Label(self.right, text="Properties", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.props = tk.Text(self.right, height=10)
        self.props.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        self.props.configure(state="disabled")

        self.prop_frame = ttk.LabelFrame(self.right, text="Edit")
        self.prop_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 10))
        self.prop_frame.columnconfigure(1, weight=1)

        self.prop_entries = {}
        self.prop_vars = {}
        self._prop_row = 0
        self.apply_btn = ttk.Button(self.prop_frame, text="Apply", command=self._apply_properties)
        self.apply_btn.grid(row=99, column=0, columnspan=2, sticky="ew", pady=(8, 2))

    def _build_data_tab(self):
        self.tab_data.columnconfigure(0, weight=1)
        self.tab_data.rowconfigure(1, weight=1)

        top = ttk.Frame(self.tab_data)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top.columnconfigure(2, weight=1)

        self.data_status = tk.StringVar(value="No run data yet. Press Start → Stop to generate a dataset.")
        ttk.Label(top, textvariable=self.data_status).grid(row=0, column=0, sticky="w")

        ttk.Button(top, text="Export Current Table…", command=self._export_current_table).grid(row=0, column=1, sticky="e", padx=(10, 0))
        ttk.Button(top, text="Export All…", command=self._export_all_tables).grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Button(top, text="Export Tracking Audit…", command=self._export_tracking_audit).grid(row=0, column=3, sticky="w", padx=(10, 0))
        ttk.Button(top, text="Export Anomaly Intel…", command=self._ai_export_report).grid(row=0, column=4, sticky="w", padx=(10, 0))

        self.data_nb = ttk.Notebook(self.tab_data)
        self.data_nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 10))

        self.data_tab_summary = ttk.Frame(self.data_nb)
        self.data_tab_veh = ttk.Frame(self.data_nb)
        self.data_tab_gps = ttk.Frame(self.data_nb)
        self.data_tab_cam = ttk.Frame(self.data_nb)
        self.data_tab_das = ttk.Frame(self.data_nb)
        self.data_tab_kalman = ttk.Frame(self.data_nb)
        self.data_tab_rmse = ttk.Frame(self.data_nb)
        self.data_tab_anom = ttk.Frame(self.data_nb)
        # Phase 2b: new "Tracks" tab, appended LAST so that the index
        # ordering of every existing tab (and therefore the export
        # dispatch mapping in _export_current_table / _export_all_tables)
        # stays exactly as before.  Display-only for now; exports for the
        # Tracks tab are intentionally deferred to Phase 6.
        self.data_tab_tracks = ttk.Frame(self.data_nb)
        # Phase 3: "Diagnostics" tab — likewise appended LAST, preserving
        # every earlier tab's notebook index.  Display-only; export is
        # deferred to Phase 6.
        self.data_tab_diag = ttk.Frame(self.data_nb)

        self.data_nb.add(self.data_tab_summary, text="Summary")
        self.data_nb.add(self.data_tab_veh, text="Vehicles")
        self.data_nb.add(self.data_tab_gps, text="GPS")
        self.data_nb.add(self.data_tab_cam, text="Cameras")
        self.data_nb.add(self.data_tab_das, text="DAS")
        self.data_nb.add(self.data_tab_kalman, text="Kalman")
        self.data_nb.add(self.data_tab_rmse, text="RMSE")
        self.data_nb.add(self.data_tab_anom, text="Issues")
        self.data_nb.add(self.data_tab_tracks, text="Tracks")
        self.data_nb.add(self.data_tab_diag, text="Diagnostics")
        # Anomaly Intelligence tab — appended last so no existing tab index shifts.
        self.data_tab_ai = ttk.Frame(self.data_nb)
        self.data_nb.add(self.data_tab_ai, text="🔬 Anomaly Intel")

        self._build_data_tables()

    def _build_data_tables(self):
        def _insert_desc_row(tv, cols, col_descs):
            """Insert a grey-italic description row as the first data row.

            Tagged "col_desc" so _clear_table skips it — the row survives
            every data reload and also appears in Excel exports automatically.
            """
            vals = tuple(col_descs.get(c, "") for c in cols)
            tv.insert("", "end", values=vals, tags=("col_desc",))
            tv.tag_configure(
                "col_desc",
                foreground="#888888",
                font=("TkDefaultFont", 9, "italic"),
            )

        def make_table(parent, cols, col_descs=None):
            """Build a Treeview+scrollbar inside *parent*.

            When col_descs is provided a grey-italic description row is
            inserted as the first row so the meaning of each column is
            always visible and survives table reloads.
            """
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(0, weight=1)
            tv = ttk.Treeview(parent, columns=cols, show="headings")
            for c in cols:
                tv.heading(c, text=c)
                w = max(80, len(c) * 8 + 24)
                tv.column(c, width=w, minwidth=w, anchor="w")
            tv.grid(row=0, column=0, sticky="nsew")
            vs = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
            vs.grid(row=0, column=1, sticky="ns")
            tv.configure(yscrollcommand=vs.set)
            if col_descs:
                _insert_desc_row(tv, cols, col_descs)
            return tv

        def make_das_table(parent, cols, col_descs=None):
            """DAS table with a simple sensor filter (All / specific sensor_id)
            and a trajectory plot below it."""
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(1, weight=2)  # table gets 2/3 of vertical space
            parent.rowconfigure(2, weight=1)  # plot gets 1/3

            # Filter bar
            bar = ttk.Frame(parent)
            bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
            ttk.Label(bar, text="Sensor:").pack(side="left")

            self._das_sensor_var = tk.StringVar(value="All")
            self._das_sensor_cb = ttk.Combobox(
                bar,
                textvariable=self._das_sensor_var,
                state="readonly",
                width=12,
                values=("All",),
            )
            self._das_sensor_cb.pack(side="left", padx=(6, 0))
            self._das_sensor_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_das_table())

            # Table
            tv = ttk.Treeview(parent, columns=cols, show="headings")
            for c in cols:
                tv.heading(c, text=c)
                w = max(80, len(c) * 8 + 24)
                tv.column(c, width=w, minwidth=w, anchor="w")
            tv.grid(row=1, column=0, sticky="nsew")
            vs = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
            vs.grid(row=1, column=1, sticky="ns")
            tv.configure(yscrollcommand=vs.set)
            if col_descs:
                _insert_desc_row(tv, cols, col_descs)

            # Trajectory plot — scrollable, one subplot per DAS fiber.
            # Wrapping in a tk.Canvas + Scrollbar lets the figure grow
            # as tall as needed (n_fibers × HEIGHT_PER_FIBER inches) without
            # every subplot squashing into a 2-inch box.
            if _MPL_OK:
                _das_plot_outer = ttk.Frame(parent)
                _das_plot_outer.grid(
                    row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 0)
                )
                _das_plot_outer.columnconfigure(0, weight=1)
                _das_plot_outer.rowconfigure(0, weight=1)

                self._das_scroll_canvas = tk.Canvas(
                    _das_plot_outer, highlightthickness=0, bd=0
                )
                _das_sb = ttk.Scrollbar(
                    _das_plot_outer, orient="vertical",
                    command=self._das_scroll_canvas.yview,
                )
                self._das_scroll_canvas.configure(yscrollcommand=_das_sb.set)
                self._das_scroll_canvas.grid(row=0, column=0, sticky="nsew")
                _das_sb.grid(row=0, column=1, sticky="ns")

                self._das_inner_frame = ttk.Frame(self._das_scroll_canvas)
                self._das_scroll_win = self._das_scroll_canvas.create_window(
                    (0, 0), window=self._das_inner_frame, anchor="nw"
                )

                # Keep the inner frame's width pinned to the visible canvas width
                def _das_canvas_resize(ev,
                                       sc=self._das_scroll_canvas,
                                       win=None):
                    win = self._das_scroll_win
                    sc.itemconfig(win, width=ev.width)
                self._das_scroll_canvas.bind("<Configure>", _das_canvas_resize)

                # Update the scroll-region whenever the inner frame changes size
                def _das_frame_configure(ev,
                                         sc=self._das_scroll_canvas):
                    sc.configure(scrollregion=sc.bbox("all"))
                self._das_inner_frame.bind("<Configure>", _das_frame_configure)

                # Mousewheel scrolling (works on Windows, macOS, Linux)
                def _das_mousewheel(ev, sc=self._das_scroll_canvas):
                    delta = int(-1 * (ev.delta / 120)) if ev.delta else (
                        -1 if ev.num == 4 else 1
                    )
                    sc.yview_scroll(delta, "units")
                self._das_scroll_canvas.bind(
                    "<MouseWheel>", _das_mousewheel
                )
                self._das_scroll_canvas.bind(
                    "<Button-4>", _das_mousewheel  # Linux scroll-up
                )
                self._das_scroll_canvas.bind(
                    "<Button-5>", _das_mousewheel  # Linux scroll-down
                )

                # Create the initial (small) matplotlib figure inside the inner frame
                self._das_fig = Figure(figsize=(6, 2.5), dpi=90)
                self._das_canvas = FigureCanvasTkAgg(
                    self._das_fig, master=self._das_inner_frame
                )
                self._das_canvas.get_tk_widget().pack(
                    fill="both", expand=True
                )
            else:
                self._das_fig = None
                self._das_canvas = None
                self._das_scroll_canvas = None

            return tv

        self.tv_summary = make_table(self.data_tab_summary, ("key", "value"))

        # Phase 8: global_track_id added to sensor and vehicle tabs.
        # Column order: tracker identity (global_track_id) comes before oracle
        # identity (vehicle_id) so the real tracking result is the first thing
        # a reader sees.  vehicle_id remains for debug / RMSE / validation.
        # Each make_table call receives col_descs so the description appears
        # directly under the column name inside the header row.
        self.tv_veh = make_table(self.data_tab_veh, (
            "t", "vehicle_id", "global_track_id", "lane_id",
            "x", "y", "v", "heading_rad",
            "a_long_mps2", "ax_world_mps2", "ay_world_mps2",
        ), col_descs={
            "t":              "timestamp (s)",
            "vehicle_id":     "oracle/sim ID",
            "global_track_id":"tracker ID",
            "lane_id":        "current lane",
            "x":              "world x (m)",
            "y":              "world y (m)",
            "v":              "speed (m/s)",
            "heading_rad":    "direction (rad)",
            "a_long_mps2":    "long. accel (m/s²)",
            "ax_world_mps2":  "world ax (m/s²)",
            "ay_world_mps2":  "world ay (m/s²)",
        })

        _sensor_acc_descs = {
            "t":              "timestamp (s)",
            "sensor_id":      "sensor ID",
            "global_track_id":"tracker ID",
            "vehicle_id":     "oracle ID (debug)",
            "x":              "measured x (m)",
            "y":              "measured y (m)",
            "speed_mps":      "speed (m/s)",
            "a_long_mps2":    "accel (m/s²)",
            "sigma_m":        "uncertainty (m)",
            "confidence":     "weight (0–1)",
        }
        self.tv_gps = make_table(self.data_tab_gps, (
            "t", "sensor_id", "global_track_id", "vehicle_id",
            "x", "y", "speed_mps", "a_long_mps2", "sigma_m", "confidence",
        ), col_descs=_sensor_acc_descs)

        self.tv_cam = make_table(self.data_tab_cam, (
            "t", "sensor_id", "global_track_id", "vehicle_id",
            "x", "y", "speed_mps", "a_long_mps2", "sigma_m", "confidence",
        ), col_descs=_sensor_acc_descs)

        self.tv_das = make_das_table(self.data_tab_das, (
            "t", "sensor_id", "global_track_id", "vehicle_id",
            "x", "y", "fiber_position_m", "speed_mps", "fiber_angle_rad", "snr",
        ), col_descs={
            "t":               "timestamp (s)",
            "sensor_id":       "sensor ID",
            "global_track_id": "tracker ID",
            "vehicle_id":      "oracle ID (debug)",
            "x":               "position x (m)",
            "y":               "position y (m)",
            "fiber_position_m":"fiber dist (m)",
            "speed_mps":       "est. speed (m/s)",
            "fiber_angle_rad": "fiber angle (rad)",
            "snr":             "signal/noise",
        })

        # Kalman tab — global_track_id already present (Phase 7).
        # Phase 6: vehicle_id_oracle is oracle-derived ground truth (from
        # the simulator), not a tracker-inferred identity.  Phase 7 added
        # the ``global_track_id`` column right before it so the user can
        # see both identities side-by-side — tracker identity first, then
        # the ground-truth reference.  The header tuple is the single
        # source of truth for column layout; build_kalman_rows_tracked
        # emits rows in this same order, and _tv_to_data(tv_kalman)
        # propagates the headers verbatim to the XLSX Kalman sheet.
        self.tv_kalman = make_table(self.data_tab_kalman, (
            "t", "global_track_id", "vehicle_id_oracle", "sources",
            "x_hat", "y_hat", "vx_hat", "vy_hat", "ax_hat", "ay_hat",
            "sigma_pos_m", "sigma_vel_mps", "pos_err_m",
        ), col_descs={
            "t":                "timestamp (s)",
            "global_track_id":  "tracker ID (primary)",
            "vehicle_id_oracle":"oracle ID (validation)",
            "sources":          "sensors used",
            "x_hat":            "est. x (m)",
            "y_hat":            "est. y (m)",
            "vx_hat":           "est. vx (m/s)",
            "vy_hat":           "est. vy (m/s)",
            "ax_hat":           "est. ax (m/s²)",
            "ay_hat":           "est. ay (m/s²)",
            "sigma_pos_m":      "pos. 1σ (m)",
            "sigma_vel_mps":    "vel. 1σ (m/s)",
            "pos_err_m":        "vs. truth (m)",
        })

        self.tv_rmse = make_table(self.data_tab_rmse, (
            "source", "samples", "rmse_x_m", "rmse_y_m", "rmse_pos_m", "notes",
        ), col_descs={
            "source":     "sensor type",
            "samples":    "# timesteps",
            "rmse_x_m":   "x-RMSE (m)",
            "rmse_y_m":   "y-RMSE (m)",
            "rmse_pos_m": "2D RMSE (m)",
            "notes":      "context",
        })

        self.tv_anom = make_table(self.data_tab_anom, ("t", "type", "details"))
        # Phase 2b: Tracks tab columns — exactly as agreed with the user.
        # ``vehicle_id_oracle`` makes it explicit that this field is the
        # ground-truth identity from the simulator, not a tracker-derived id.
        self.tv_tracks = make_table(self.data_tab_tracks, (
            "t", "global_track_id", "vehicle_id_oracle", "source",
            "x", "y", "v", "segment_id", "lane_id", "tentative",
            "n_gps", "n_cam", "n_das", "n_state",
        ), col_descs={
            "t":                "timestamp (s)",
            "global_track_id":  "tracker ID",
            "vehicle_id_oracle":"oracle ID",
            "source":           "event source",
            "x":                "position x (m)",
            "y":                "position y (m)",
            "v":                "speed (m/s)",
            "segment_id":       "road segment",
            "lane_id":          "current lane",
            "tentative":        "not yet confirmed",
            "n_gps":            "# GPS hits",
            "n_cam":            "# camera hits",
            "n_das":            "# DAS hits",
            "n_state":          "# state updates",
        })

        # Phase 3: Diagnostics tab — two stacked Treeviews.
        # Top pane: manager-level summary (key/value).
        # Bottom pane: per-track diagnostic row table.
        # Column order below MUST match the order returned by
        # ``simstudio.tracking.TrackManager.DIAGNOSTIC_COLUMNS``.
        self._build_diag_tab(self.data_tab_diag)
        # Anomaly Intelligence tab — built last; never raises (broad guard below).
        try:
            self._build_anomaly_intel_tab(self.data_tab_ai)
        except Exception as _ai_err:
            import traceback as _tb
            print(f"[AnomalyIntel] build failed: {_ai_err}\n{_tb.format_exc()}")

    def _build_diag_tab(self, parent):
        """Build the Phase 3 Tracking-Diagnostics tab.

        The tab is split vertically into two Treeview panes:

        * **Top (summary)** — two-column ``(key, value)`` view of the
          manager-level counters returned by
          ``TrackManager.summary()``.  Small, fixed-width, no scroll.
        * **Bottom (per-track)** — one row per track, using exactly the
          columns declared in ``TrackManager.DIAGNOSTIC_COLUMNS`` so the
          contract is owned by the tracker module (the GUI follows).

        This method only builds widgets; it does not populate them.  The
        populate step lives in ``_populate_diag_tab`` and runs at the
        same cadence as the other data tables.
        """
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)   # summary pane: fixed height
        parent.rowconfigure(1, weight=1)   # per-track pane: fills remaining space

        # --- Top summary pane --------------------------------------------
        summary_frame = ttk.LabelFrame(parent, text="Tracker summary")
        summary_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 4))
        summary_frame.columnconfigure(0, weight=1)

        self.tv_diag_summary = ttk.Treeview(
            summary_frame, columns=("key", "value"), show="headings", height=6,
        )
        for c, w in (("key", 220), ("value", 160)):
            self.tv_diag_summary.heading(c, text=c)
            self.tv_diag_summary.column(c, width=w, anchor="w")
        self.tv_diag_summary.grid(row=0, column=0, sticky="ew")

        # --- Bottom per-track pane --------------------------------------
        # Columns are read from the tracker module so there is exactly one
        # source of truth.  Importing lazily avoids a hard dependency if
        # the tracking module failed to import at startup.
        try:
            from ..tracking import TrackManager as _TM
            cols = tuple(_TM.DIAGNOSTIC_COLUMNS)
        except Exception:
            cols = (
                "global_track_id", "vehicle_id_oracle", "segment_id", "state",
                "t_born", "t_last", "duration_s",
                "n_gps", "n_cam", "n_das", "n_state",
                "n_segments", "segments_visited",
                "hypothesis_count", "id_switches", "n_updates_total",
            )

        diag_frame = ttk.Frame(parent)
        diag_frame.grid(row=1, column=0, sticky="nsew")
        diag_frame.columnconfigure(0, weight=1)
        diag_frame.rowconfigure(0, weight=1)

        _diag_descs = {
            "global_track_id":  "tracker ID",
            "vehicle_id_oracle":"oracle ID (validation)",
            "segment_id":       "current segment",
            "state":            "track state",
            "t_born":           "sim time at birth (s)",
            "t_last":           "sim time, last update (s)",
            "duration_s":       "track age (s)",
            "n_gps":            "# GPS hits",
            "n_cam":            "# camera hits",
            "n_das":            "# DAS hits",
            "n_state":          "# state updates",
            "n_segments":       "segments visited",
            "segments_visited": "segment ID list",
            "hypothesis_count": "# vehicle hypotheses",
            "id_switches":      "# ID switches",
            "n_updates_total":  "total updates",
        }
        self.tv_diag = ttk.Treeview(diag_frame, columns=cols, show="headings")
        for c in cols:
            self.tv_diag.heading(c, text=c)
            w = 220 if c == "segments_visited" else 130
            self.tv_diag.column(c, width=w, anchor="w")
        self.tv_diag.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(diag_frame, orient="vertical", command=self.tv_diag.yview)
        vs.grid(row=0, column=1, sticky="ns")
        self.tv_diag.configure(yscrollcommand=vs.set)
        # Description row — grey italic, preserved across clears.
        desc_vals = tuple(_diag_descs.get(c, "") for c in cols)
        self.tv_diag.insert("", "end", values=desc_vals, tags=("col_desc",))
        self.tv_diag.tag_configure(
            "col_desc", foreground="#888888", font=("TkDefaultFont", 9, "italic"),
        )

    # ---------------- Menu + shortcuts ----------------
    def _build_menu(self):
        m = tk.Menu(self)

        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="New Scene", command=self._new_scene, accelerator="Cmd/Ctrl+N")
        fm.add_command(label="Create Demo Scene", command=self._create_demo_scene)
        fm.add_separator()
        fm.add_command(label="Load Scene…", command=self._load_scene, accelerator="Cmd/Ctrl+O")
        fm.add_command(label="Save Scene…", command=self._save_scene, accelerator="Cmd/Ctrl+S")
        fm.add_separator()
        fm.add_command(label="Export Current Table…", command=self._export_current_table)
        fm.add_command(label="Export All…", command=self._export_all_tables)
        fm.add_command(label="Export Tracking Audit…", command=self._export_tracking_audit)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=fm)

        em = tk.Menu(m, tearoff=0)
        em.add_command(label="Undo", command=self._undo, accelerator="Cmd/Ctrl+Z")
        em.add_command(label="Redo", command=self._redo, accelerator="Cmd/Ctrl+Shift+Z")
        em.add_separator()
        em.add_command(label="Delete Selection", command=self._delete_selection, accelerator="Del")
        m.add_cascade(label="Edit", menu=em)

        vm = tk.Menu(m, tearoff=0)
        vm.add_command(label="Simulation Tab", command=lambda: self.nb.select(self.tab_sim))
        vm.add_command(label="Data Tab", command=lambda: self.nb.select(self.tab_data))
        m.add_cascade(label="View", menu=vm)

        self.config(menu=m)

    def _bind_shortcuts(self):
        def bind_seq(seq, fn):
            self.bind_all(seq, lambda e: fn())

        bind_seq("<Command-n>", self._new_scene)
        bind_seq("<Control-n>", self._new_scene)
        bind_seq("<Command-o>", self._load_scene)
        bind_seq("<Control-o>", self._load_scene)
        bind_seq("<Command-s>", self._save_scene)
        bind_seq("<Control-s>", self._save_scene)

        bind_seq("<Command-z>", self._undo)
        bind_seq("<Control-z>", self._undo)
        bind_seq("<Command-Shift-Z>", self._redo)
        bind_seq("<Control-Shift-Z>", self._redo)
        bind_seq("<Command-y>", self._redo)
        bind_seq("<Control-y>", self._redo)

        self.bind_all("<Delete>", lambda e: self._delete_selection())
        self.bind_all("<BackSpace>", lambda e: self._delete_selection())

        self.bind_all("v", lambda e: self._set_tool(TOOL_SELECT))
        self.bind_all("s", lambda e: self._set_tool(TOOL_SEGMENT))
        self.bind_all("r", lambda e: self._set_tool(TOOL_VEHICLE))
        self.bind_all("g", lambda e: self._set_tool(TOOL_GPS))
        self.bind_all("c", lambda e: self._set_tool(TOOL_CAMERA))
        self.bind_all("d", lambda e: self._set_tool(TOOL_DAS))
        self.bind_all("x", lambda e: self._set_tool(TOOL_BOUNDARY_EXIT))
        self.bind_all("m", lambda e: self._import_roads_from_map())

        # route planning
        self.bind_all("p", lambda e: self._begin_plan_route())
        self.bind_all("<Return>", lambda e: self._commit_pending())
        self.bind_all("<Escape>", lambda e: self._cancel_pending())

    def _set_tool(self, t):
        self.tool.set(t)
        self._tool_changed()

    # ---------------- Scene ops ----------------
    def _reset_runtime(self):
        self._all_events = []
        self._persistent_collided_ids = set()
        self._persistent_route_complete_ids = set()
        self._persistent_stuck_positions = []
        with getattr(self, '_snap_lock', threading.Lock()):
            self._snaps = []
            self._playhead = -1
        self._playback_mode = False
        self._dirty_from_scrub = False
        self._vehicle_routes = {}
        self._route_progress = {}
        self._time_max = 0.0
        self.view_time = None
        self.time_var.set(0.0)
        try:
            self.time_scale.configure(to=0.0)
        except Exception:
            pass
        self._data_ready = False
        try:
            self._open_end_nodes.clear()
        except Exception:
            pass
        self._clear_table(self.tv_summary)
        self._clear_table(self.tv_veh)
        self._clear_table(self.tv_gps)
        self._clear_table(self.tv_cam)
        self._clear_table(self.tv_das)
        self._clear_table(self.tv_kalman)
        self._clear_table(self.tv_rmse)
        self._clear_table(self.tv_anom)
        # Phase 2b: keep the Tracks tab in lock-step with the other data
        # tables during a full data-reset (new scene / clear dataset).
        try:
            self._clear_table(self.tv_tracks)
        except Exception:
            pass
        # Phase 3: also clear the Tracking-Diagnostics tab so the GUI
        # cannot display stale counters after a dataset clear.
        try:
            tv_sum = getattr(self, "tv_diag_summary", None)
            if tv_sum is not None:
                self._clear_table(tv_sum)
        except Exception:
            pass
        try:
            tv_rows = getattr(self, "tv_diag", None)
            if tv_rows is not None:
                self._clear_table(tv_rows)
        except Exception:
            pass
        try:
            if hasattr(self, "_track_buffer"):
                self._track_buffer.clear()
        except Exception:
            pass
        try:
            if getattr(self, "_track_manager", None) is not None:
                self._track_manager.reset()
        except Exception:
            pass
        self._das_rows_full = []
        try:
            if hasattr(self, "_das_sensor_var"):
                self._das_sensor_var.set("All")
            if hasattr(self, "_das_sensor_cb"):
                self._das_sensor_cb.configure(values=("All",))
        except Exception:
            pass
        self.data_status.set("No run data yet. Press Start → Stop to generate a dataset.")

        # ── Clear Anomaly Intel tab ───────────────────────────────────────────
        # Reset all result state so the next simulation starts clean.
        self._ai_analysis_result = None
        try:
            # Clear the vehicle score treeview
            for iid in self.tv_ai_scores.get_children():
                self.tv_ai_scores.delete(iid)
        except Exception:
            pass
        try:
            self._ai_status.set("Ready — click Analyze after running a simulation.")
        except Exception:
            pass
        try:
            self._ai_verdict_var.set("")
        except Exception:
            pass
        try:
            self._ai_export_btn.configure(state="disabled")
        except Exception:
            pass
        try:
            for iid in self._ai_dist_table.get_children():
                self._ai_dist_table.delete(iid)
        except Exception:
            pass
        # Clear all four chart canvases
        for fig_attr, canvas_attr in [
            ("_ai_fig_heat",  "_ai_canvas_heat"),
            ("_ai_fig_bars",  "_ai_canvas_bars"),
            ("_ai_fig_dist",  "_ai_canvas_dist"),
            ("_ai_fig_cusum", "_ai_canvas_cusum"),
        ]:
            try:
                fig = getattr(self, fig_attr, None)
                canvas = getattr(self, canvas_attr, None)
                if fig is not None:
                    fig.clear()
                if canvas is not None:
                    canvas.draw()
            except Exception:
                pass

    def _new_scene(self):
        self._stop(silent=True, skip_finalize=True)
        self.vp.reset()
        self.world = World()
        self.sim = Simulation(self.bus, self.world)
        self.sim.rebuild_lanes()
        self.selected = None
        self.pending = None
        self.undo = UndoRedo()
        self.undo.push(self.world)
        self._reset_runtime()
        self._update_properties()
        self._scene_name = "Untitled"
        self._update_window_title()
        self._render()
        self.status.set("New empty scene. Add segments to begin.")

    def _create_demo_scene(self):
        self._stop(silent=True, skip_finalize=True)
        self.vp.reset()
        self.world = World()
        self.sim = Simulation(self.bus, self.world)

        # Demo: quick sanity test scene (so users can test immediately)
        self.world.nodes["n1"] = Node(id="n1", x=-420, y=0)
        self.world.nodes["n2"] = Node(id="n2", x=0, y=0)
        self.world.nodes["n3"] = Node(id="n3", x=420, y=0)
        self.world.nodes["n4"] = Node(id="n4", x=0, y=250)

        # Demo uses the current road model: single-lane, one-way segments only.
        self.world.segments["seg1"] = Segment(id="seg1", n0="n1", n1="n2", lanes=1, one_way=True)
        self.world.segments["seg2"] = Segment(id="seg2", n0="n2", n1="n3", lanes=1, one_way=True)
        self.world.segments["seg3"] = Segment(id="seg3", n0="n2", n1="n4", lanes=1, one_way=True)

        self.sim.rebuild_lanes()
        any_lane = next(iter(self.world.lanes.keys()), None)
        if any_lane:
            profile = _random_vehicle_macro_profile(float(getattr(self.world.segments.get(self.world.lanes[any_lane].segment_id), "speed_limit_mps", 13.9) or 13.9))
            self.world.vehicles["veh1"] = Vehicle(id="veh1", lane_id=any_lane, s=0.0, v=13.0, weight_kg=1500.0, **profile)
        self.world.gps["gps1"] = GPSSensor(id="gps1", x=-120, y=120, radius_m=60.0)
        self.world.cameras["cam1"] = CameraSensor(id="cam1", x=0, y=-220, heading_rad=math.pi/2, fov_deg=70.0, range_m=260.0)
        self.world.das["das1"] = DASSensor(id="das1", segment_id="seg1")

        self.selected = None
        self.pending = None
        self.undo = UndoRedo()
        self.undo.push(self.world)
        self._reset_runtime()
        self._update_properties()
        self._scene_name = "Demo Scene"
        self._update_window_title()
        self._render()
        self.status.set("Created demo scene (for quick testing). Press Start.")

    def _random_populate(self):
        """Populate the current scene with a realistic random mix of vehicles & sensors."""
        if not self.world.nodes or not self.world.segments:
            messagebox.showinfo("Random Populate", "Create a scene first (nodes + segments).")
            return

        # Ensure lane topology exists.
        try:
            self.sim.rebuild_lanes()
        except Exception as ex:
            messagebox.showerror("Random Populate", f"Failed to build lanes: {ex}")
            return

        # IMPORTANT: this window must be *modeless*.
        # Users need to click on the main canvas to tag traffic segments.
        win = tk.Toplevel(self)
        win.title("Random Populate")
        def _on_close_random():
            try:
                _stop_mark()
            except Exception:
                pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close_random)
        win.bind("<Escape>", lambda _e: _stop_mark())

        win.resizable(False, False)
        frm = ttk.Frame(win, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        seed_var = tk.StringVar(value="")
        veh_var = tk.StringVar(value="")  # "N" or "min-max"
        gps_var = tk.StringVar(value="")
        cam_var = tk.StringVar(value="")
        das_var = tk.StringVar(value="")
        clear_var = tk.BooleanVar(value=True)

        # Vehicle distributions
        v_dist_kind = tk.StringVar(value="gauss")
        v_min = tk.StringVar(value="20")
        v_max = tk.StringVar(value="100")
        v_mean = tk.StringVar(value="55")
        v_std = tk.StringVar(value="13")

        w_dist_kind = tk.StringVar(value="gauss")
        w_min = tk.StringVar(value="800")
        w_max = tk.StringVar(value="36000")
        w_mean = tk.StringVar(value="1600")
        w_std = tk.StringVar(value="500")

        # Vehicles are modeled as points -> no footprint distributions.

        spawn_var = tk.BooleanVar(value=False)
        spawn_rate = tk.StringVar(value="10")  # vehicles per minute
        spawn_max = tk.StringVar(value="")

        ttk.Label(frm, text="Seed (optional):").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=seed_var, width=18).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="Vehicles (N or min-max):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=veh_var, width=18).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="GPS sensors (optional):").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=gps_var, width=18).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="Cameras (optional):").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=cam_var, width=18).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="DAS sensors (optional):").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=das_var, width=18).grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(frm, text="Clear existing vehicles/sensors", variable=clear_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 8))

        # Vehicle distribution controls
        dist = ttk.LabelFrame(frm, text="Vehicle distributions (properties)")
        dist.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        for j in range(1, 6):
            dist.columnconfigure(j, weight=1)

        ttk.Label(dist, text="Property").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(dist, text="Kind").grid(row=0, column=1, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(dist, text="Min").grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(dist, text="Max").grid(row=0, column=3, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(dist, text="Mean").grid(row=0, column=4, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(dist, text="Std").grid(row=0, column=5, sticky="w", padx=6, pady=(6, 2))

        def _dist_row(r, name, kind_var, min_var, max_var, mean_var, std_var, suffix=""):
            ttk.Label(dist, text=name + suffix).grid(row=r, column=0, sticky="w", padx=6, pady=2)
            cb = ttk.Combobox(dist, textvariable=kind_var, values=["gauss", "uniform"], width=8, state="readonly")
            cb.grid(row=r, column=1, sticky="ew", padx=6, pady=2)
            ttk.Entry(dist, textvariable=min_var, width=8).grid(row=r, column=2, sticky="ew", padx=6, pady=2)
            ttk.Entry(dist, textvariable=max_var, width=8).grid(row=r, column=3, sticky="ew", padx=6, pady=2)
            ttk.Entry(dist, textvariable=mean_var, width=8).grid(row=r, column=4, sticky="ew", padx=6, pady=2)
            ttk.Entry(dist, textvariable=std_var, width=8).grid(row=r, column=5, sticky="ew", padx=6, pady=2)

        _dist_row(1, "Speed", v_dist_kind, v_min, v_max, v_mean, v_std, " (km/h)")
        _dist_row(2, "Weight", w_dist_kind, w_min, w_max, w_mean, w_std, " (kg)")

        # Continuous spawning controls
        spf = ttk.LabelFrame(frm, text="Vehicles over time")
        spf.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        spf.columnconfigure(1, weight=1)
        ttk.Checkbutton(spf, text="Spawn vehicles continuously while running", variable=spawn_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 4)
        )
        ttk.Label(spf, text="Rate (vehicles/min):").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(spf, textvariable=spawn_rate, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(spf, text="Max total (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=(2, 6))
        ttk.Entry(spf, textvariable=spawn_max, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=(2, 6))

        msg = tk.StringVar(value="")
        # Traffic load annotation (per-segment)
        traffic = ttk.LabelFrame(frm, text="Traffic load (tag segments)")
        traffic.grid(row=8, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(
            traffic,
            text="Click a level, then click road segments to toggle. Click Done when finished.",
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        def _start_mark(level: str):
            # Enable explicit tagging mode (canvas clicks are consumed by segment tagging).
            self._traffic_mark_active = True
            self._traffic_mark_level = level
            try:
                self.canvas.config(cursor="crosshair")
            except Exception:
                pass
            self.status.set(f"Traffic TAG MODE [{level.upper()}]: click segments to tag/untag. Click again to remove. Done/Esc to exit.")

        def _stop_mark():
            # Exit tagging mode and return to normal selection behavior.
            self._traffic_mark_active = False
            self._traffic_mark_level = None
            try:
                self.canvas.config(cursor="")
            except Exception:
                pass
            self.status.set("Traffic tagging: off.")

        ttk.Button(traffic, text="Light", command=lambda: _start_mark("light")).grid(row=1, column=0, padx=(0, 6), pady=(6, 0))
        ttk.Button(traffic, text="Medium", command=lambda: _start_mark("medium")).grid(row=1, column=1, padx=(0, 6), pady=(6, 0))
        ttk.Button(traffic, text="Heavy", command=lambda: _start_mark("heavy")).grid(row=1, column=2, padx=(0, 6), pady=(6, 0))
        ttk.Button(traffic, text="Done", command=_stop_mark).grid(row=1, column=3, pady=(6, 0))

        ttk.Label(frm, textvariable=msg, foreground="#cfcfcf").grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def _parse_int(s: str) -> Optional[int]:
            s = (s or "").strip()
            if not s:
                return None
            try:
                return int(s)
            except Exception:
                return None

        def _parse_range(s: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
            """Return (exact, min, max) from a string: "N" or "min-max"."""
            s = (s or "").strip()
            if not s:
                return (None, None, None)
            if "-" in s:
                parts = [p.strip() for p in s.split("-", 1)]
                a = _parse_int(parts[0])
                b = _parse_int(parts[1])
                return (None, a, b)
            return (_parse_int(s), None, None)

        def _dist_dict(kind_var, min_var, max_var, mean_var, std_var) -> dict:
            def f(x, default=None):
                try:
                    return float(x)
                except Exception:
                    return default
            return {
                "kind": (kind_var.get() or "gauss").strip().lower(),
                "min": f(min_var.get(), None),
                "max": f(max_var.get(), None),
                "mean": f(mean_var.get(), None),
                "std": f(std_var.get(), None),
            }

        def do_populate():
            try:
                _stop_mark()

                seed = _parse_int(seed_var.get())
                veh_exact, veh_min, veh_max = _parse_range(veh_var.get())
                gps_exact, gps_min, gps_max = _parse_range(gps_var.get())
                cam_exact, cam_min, cam_max = _parse_range(cam_var.get())
                das_exact, das_min, das_max = _parse_range(das_var.get())

                spec = RandomizeSpec(
                    seed=seed,
                    n_vehicles=veh_exact,
                    vehicles_min=veh_min,
                    vehicles_max=veh_max,

                    n_gps=gps_exact,
                    gps_min=gps_min,
                    gps_max=gps_max,

                    n_cameras=cam_exact,
                    cameras_min=cam_min,
                    cameras_max=cam_max,

                    n_das=das_exact,
                    das_min=das_min,
                    das_max=das_max,

                    vehicle_speed_kmh=_dist_dict(v_dist_kind, v_min, v_max, v_mean, v_std),
                    vehicle_weight_kg=_dist_dict(w_dist_kind, w_min, w_max, w_mean, w_std),

                    spawn_enabled=bool(spawn_var.get()),
                    spawn_rate_vpm=float(_parse_int(spawn_rate.get()) or 0),
                    spawn_max_total=_parse_int(spawn_max.get()),
                    clear_existing=bool(clear_var.get()),
                )

                # Push undo snapshot before mutation.
                self.undo.push(self.world)

                counts = randomize_world(self.world, getattr(self.sim, "_lane_meta", {}), spec)

                # If the user tagged segments with traffic levels, immediately enforce
                # density targets so they can see the effect *before* pressing Start.
                try:
                    self.sim._enforce_segment_traffic()
                except Exception:
                    pass

                # Save continuous spawning configuration for the run loop
                if spec.spawn_enabled and float(spec.spawn_rate_vpm or 0.0) > 0.0:
                    self._rand_spawn_spec = spec
                    self._rand_spawn_state = {"acc": 0.0}
                else:
                    self._rand_spawn_spec = None
                    self._rand_spawn_state = {"acc": 0.0}

                self._reset_runtime()
                self._update_properties()
                self._render()
                self.status.set(
                    f"Random populated: vehicles={counts['vehicles']}, gps={counts['gps']}, cameras={counts['cameras']}, das={counts['das']}"
                )
                win.destroy()
            except Exception as ex:
                msg.set(f"Error: {ex}")

        def _close_win():
            # Always exit traffic-tagging mode when closing the dialog.
            try:
                self._traffic_mark_active = False
                self._traffic_mark_level = None
                self.canvas.config(cursor="")
            except Exception:
                pass
            win.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=10, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=_close_win).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Populate", command=do_populate).grid(row=0, column=1)

        win.protocol("WM_DELETE_WINDOW", _close_win)

        win.transient(self)
        # Do NOT call grab_set(): it blocks interacting with the main canvas on many platforms.
        win.focus_force()

    def _normalize_single_lane(self) -> None:
        """Project-wide constraint: one lane per direction.

        Enforces lanes=1 on every segment so the engine always operates in
        one-lane-per-direction mode.  one_way is intentionally left untouched
        so that bidirectional segments (one_way=False) survive load and import
        and produce both a forward and a backward lane in rebuild_lanes().
        """
        for seg in self.world.segments.values():
            try:
                seg.lanes = 1
            except Exception:
                pass

    def _randomize_selected_vehicle_profile(self):
        if not self.selected or self.selected[0] != "vehicle":
            return
        sid = str(self.selected[1])
        veh = self.world.vehicles.get(sid)
        if veh is None:
            return
        lane = self.world.lanes.get(veh.lane_id)
        seg = self.world.segments.get(lane.segment_id) if lane is not None else None
        speed_limit = float(getattr(seg, "speed_limit_mps", 13.9) or 13.9)
        profile = _random_vehicle_macro_profile(speed_limit)
        for k, val in profile.items():
            setattr(veh, k, val)
        veh.v = max(float(profile.get("speed_min_mps", 0.0) or 0.0), min(float(profile.get("target_speed_mps", veh.v) or veh.v), float(profile.get("speed_max_mps", speed_limit) or speed_limit)))
        veh.a_long_mps2 = 0.0
        try:
            self.sim.apply_vehicle_macro_profile(sid)
        except Exception:
            pass
        self._update_properties()
        self._render()
        self.status.set(f"Vehicle {sid} got a new realistic profile.")

    def _save_scene(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".sim.json",
            filetypes=[("SimStudio Scene", "*.sim.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffixes[-2:] != [".sim", ".json"]:
            out_path = out_path.with_suffix("").with_suffix("").with_name(out_path.stem.split(".")[0] + ".sim.json")
        save_world(out_path, self.world)
        # Update the scene name to match the just-saved filename
        _stem = out_path.name
        for _ext in (".sim.json", ".json"):
            if _stem.endswith(_ext):
                _stem = _stem[: -len(_ext)]
                break
        self._scene_name = _stem or self._scene_name
        self._update_window_title()
        self._render()
        self.status.set(f"Saved scene: {out_path}")

    def _load_scene(self):
        path = filedialog.askopenfilename(filetypes=[("SimStudio Scene", "*.sim.json"), ("JSON", "*.json")])
        if not path:
            return
        # Snapshot current state so we can roll back if loading fails.
        _prev_world = self.world
        _prev_sim = self.sim
        try:
            self._stop(silent=True, skip_finalize=True)
            self.world = load_world(Path(path))
            self._normalize_single_lane()
            self.sim = Simulation(self.bus, self.world)
            self.sim.rebuild_lanes()
            self.selected = None
            self.pending = None
            self.undo = UndoRedo()
            self.undo.push(self.world)
            self._reset_runtime()
            self._update_properties()
            # Derive a clean scene name from the filename (strip .sim.json / .json)
            _p = Path(path)
            _stem = _p.name
            for _ext in (".sim.json", ".json"):
                if _stem.endswith(_ext):
                    _stem = _stem[: -len(_ext)]
                    break
            self._scene_name = _stem or "Untitled"
            self._scene_json_path = Path(path)
            self._update_window_title()
            # Fit the viewport to the loaded content.  If the canvas has not yet
            # been laid out (winfo_width returns 1 on first open), defer one frame
            # so Tk can assign real dimensions before we compute the scale.
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                self.vp.fit_to_world(self.world, cw, ch)
            else:
                def _deferred_fit():
                    self.vp.fit_to_world(
                        self.world,
                        self.canvas.winfo_width() or 900,
                        self.canvas.winfo_height() or 600,
                    )
                    self._render()
                self.after(50, _deferred_fit)
            self._render()
            self.status.set(f"Loaded scene: {path}")
        except Exception as ex:
            import traceback
            # Restore the previous working state so the GUI stays usable.
            self.world = _prev_world
            self.sim = _prev_sim
            # Print full traceback to stderr for debugging.
            traceback.print_exc()
            # Build a user-friendly message that names the actual problem.
            detail = str(ex)
            messagebox.showerror(
                "Load Scene Failed",
                f"Could not load file:\n{Path(path).name}\n\n"
                f"Reason: {detail}\n\n"
                "Common causes:\n"
                "  • 'segments'/'vehicles'/'das' are lists — must be dicts keyed by ID\n"
                "  • Vehicle uses 'segment_id' instead of 'lane_id'\n"
                "  • Vehicle uses 'mass_kg'/'s_m' instead of 'weight_kg'/'s'\n"
                "  • 'lanes' section is missing\n"
                "  • Unknown field name in a sensor or vehicle\n\n"
                "See console for the full error traceback.",
            )
            self.status.set(f"Load failed — {detail}")
            try:
                self._render()
            except Exception:
                pass

    def _import_roads_from_map(self):
        # Open a modal map window to select an area and import roads (two-way only).
        try:
            win = MapImportWindow(self)
            win.transient(self)
            win.grab_set()
        except Exception as ex:
            messagebox.showerror('Import Roads', f'Failed to open map window: {ex}')


    # ---------------- Run ----------------
    def _start(self):

        # Ensure no UI is left in traffic-tagging mode.
        self._traffic_mark_active = False
        self._traffic_mark_level = None
        try:
            self.canvas.config(cursor="")
        except Exception:
            pass
        if self._running:
            return
        # Timeline is for review only: you can resume only from the end.
        if self._review_mode:
            messagebox.showerror('Timeline', 'You are viewing the past. Move the timeline to the end to resume playback.')
            return
        self._data_ready = False

        # Phase 2b: each Start begins cleanly from the tracker's point of
        # view — discard accumulated tracks, ring-buffered track events,
        # and the displayed Tracks tab.  We do this BEFORE any simulation
        # thread is resumed so races cannot put stale events into the
        # freshly-cleared buffer.  ``reset()`` is a no-op when the tracker
        # was never attached (e.g. import failed at startup).
        try:
            if getattr(self, "_track_manager", None) is not None:
                self._track_manager.reset()
        except Exception:
            pass
        try:
            if hasattr(self, "_track_buffer"):
                self._track_buffer.clear()
        except Exception:
            pass
        try:
            self._clear_table(self.tv_tracks)
        except Exception:
            pass
        # Phase 3: reset the Tracking-Diagnostics tab in lock-step.  These
        # widgets may not exist if _build_diag_tab failed; guard with
        # getattr and swallow every exception so Start never fails here.
        try:
            tv_sum = getattr(self, "tv_diag_summary", None)
            if tv_sum is not None:
                self._clear_table(tv_sum)
        except Exception:
            pass
        try:
            tv_rows = getattr(self, "tv_diag", None)
            if tv_rows is not None:
                self._clear_table(tv_rows)
        except Exception:
            pass
        try:
            self._open_end_nodes.clear()
        except Exception:
            pass
        # After playback starts, editing undo is disabled to avoid resetting time/state.
        self._ever_played = True
        self.undo.clear()
        # If starting fresh (no timeline yet), reset dataset buffers
        with self._snap_lock:
            has_history = len(self._snaps) > 0
        if not has_history:
            self._all_events = []
            self._persistent_collided_ids = set()
            self._persistent_route_complete_ids = set()
            self._persistent_stuck_positions = []
            self._time_max = 0.0
            self.view_time = None
            self.time_var.set(0.0)
            self._vehicle_routes = {}
            self._route_progress = {}
            try:
                self.time_scale.configure(to=0.0)
            except Exception:
                pass
        else:
            # Resume from current playhead/time.
            # If the user scrubbed back, we treat it as a true "time travel":
            # drop the future history and continue simulation from that moment.
            if self._dirty_from_scrub:
                t0 = float(self.time_var.get())
                with self._snap_lock:
                    if self._playhead >= 0 and self._playhead < len(self._snaps) - 1:
                        self._snaps = self._snaps[: self._playhead + 1]
                # Drop future events beyond the chosen time
                kept = []
                for ev in self._all_events:
                    p = getattr(ev, 'payload', None) or {}
                    tt = p.get('t', None)
                    if tt is None or float(tt) <= t0 + 1e-9:
                        kept.append(ev)
                self._all_events = kept
                # Rebuild persistent marker sets from the kept (pre-scrub) events
                # so markers correctly reflect only the surviving history.
                self._persistent_collided_ids = set()
                self._persistent_route_complete_ids = set()
                self._persistent_stuck_positions = []
                for _kev in kept:
                    _kt = getattr(_kev, 'topic', '')
                    _kp = getattr(_kev, 'payload', None) or {}
                    if _kt == 'world.collision':
                        for _vid in _kp.get('vehicle_ids', []) or []:
                            self._persistent_collided_ids.add(str(_vid))
                    elif _kt == 'world.route_complete':
                        _vid = _kp.get('vehicle_id')
                        if _vid:
                            self._persistent_route_complete_ids.add(str(_vid))
                    elif _kt == 'world.vehicle_stuck':
                        _x, _y = _kp.get('x'), _kp.get('y')
                        if _x is not None and _y is not None:
                            _pos = (float(_x), float(_y))
                            if _pos not in self._persistent_stuck_positions:
                                self._persistent_stuck_positions.append(_pos)
                self._time_max = t0
                try:
                    self.time_scale.configure(to=self._time_max)
                except Exception:
                    pass
                self._playback_mode = False
                self._dirty_from_scrub = False
            else:
                # If no edits and we are not scrubbing, we can replay snapshots quickly.
                self._playback_mode = (self._playhead >= 0 and self._playhead < len(self._snaps) - 1)
        self._clear_table(self.tv_anom)
        self.data_status.set('Running… Press Stop to finalize dataset.')
        self.status.set('Running…')
        self._running = True
        self._last_tick = time.time()
        threading.Thread(target=self._sim_loop, daemon=True).start()

    def _stop(self, silent=False, skip_finalize=False):

        # Always exit traffic-tagging mode on stop (prevents clicks from tagging segments).
        self._traffic_mark_active = False
        self._traffic_mark_level = None
        try:
            self.canvas.config(cursor="")
        except Exception:
            pass
        if not self._running:
            if not silent:
                self.status.set("Stopped.")
            return
        self._running = False
        # After stopping, you are at the end of the timeline.
        self._review_mode = False
        self.view_time = None
        try:
            self.time_var.set(self._time_max)
        except Exception:
            pass
        self.view_time = float(self.time_var.get())
        if not silent:
            self.status.set("Stopped. Dataset ready in Data tab.")
        if not skip_finalize:
            self._finalize_dataset()

    def _step_once(self):
        self.sim.step(0.02)

    def _sim_loop(self):
        while self._running:
            now = time.time()
            dt = now - (self._last_tick or now)
            self._last_tick = now
            dt = max(0.0, min(0.03, dt))

            # If we resumed from the past and didn't change anything, replay existing snapshots until we catch up.
            if self._playback_mode:
                with self._snap_lock:
                    if self._playhead < len(self._snaps) - 1:
                        self._playhead += 1
                        snap = self._snaps[self._playhead]
                    else:
                        snap = None
                        self._playback_mode = False
                if snap is not None:
                    try:
                        self.sim.restore(snap)
                        self.world = self.sim.world
                        self.time_var.set(float(snap.get('t', 0.0)))
                    except Exception:
                        pass
                    time.sleep(0.01)
                    continue

            # Live stepping (or recompute after edits)
            self.sim.step(dt)

            # Optional continuous vehicle spawning
            try:
                if self._rand_spawn_spec is not None:
                    spawned = spawn_vehicles_over_time(
                        self.world,
                        getattr(self.sim, "_lane_meta", {}) or {},
                        self._rand_spawn_spec,
                        float(dt),
                        self._rand_spawn_state,
                    )
                    if spawned:
                        # Lanes don't change, but UI should refresh when cars appear.
                        self._request_render()
            except Exception:
                pass

            # Record snapshot for timeline
            try:
                snap = self.sim.snapshot()
                with self._snap_lock:
                    # If user scrubbed back and then edited, drop future and append new truth
                    if self._dirty_from_scrub and self._playhead >= 0 and self._playhead < len(self._snaps) - 1:
                        self._snaps = self._snaps[: self._playhead + 1]
                    self._snaps.append(snap)
                    self._playhead = len(self._snaps) - 1
                    self._dirty_from_scrub = False
            except Exception:
                pass

            time.sleep(0.01)
    def _undo(self):
        if self._ever_played:
            messagebox.showinfo('Undo', 'Undo is available only before pressing Start. Use the timeline slider to inspect the past.')
            return
        w = self.undo.do_undo(self.world)
        if w is None:
            return
        self.world = w
        self.sim = Simulation(self.bus, self.world)
        self.sim.rebuild_lanes()
        self.selected = None
        self.pending = None
        self._render()
        self._update_properties()
        self.status.set("Undo.")

    def _redo(self):
        w = self.undo.do_redo(self.world)
        if w is None:
            return
        self.world = w
        self.sim = Simulation(self.bus, self.world)
        self.sim.rebuild_lanes()
        self.selected = None
        self.pending = None
        self._render()
        self._update_properties()
        self.status.set("Redo.")

    # ---------------- Tools ----------------
    def _tool_changed(self):
        self.pending = None

        # Segment menu: only show when the Add Segment tool is active
        try:
            if self.tool.get() == TOOL_SEGMENT:
                self.seg_menu.grid()
            else:
                self.seg_menu.grid_remove()
        except Exception:
            pass

        t = self.tool.get()
        if t == TOOL_BOUNDARY_EXIT:
            self.status.set(
                "Exit Node mode: click a node to toggle it as a route endpoint. "
                "Vehicles reaching an exit node leave the map cleanly (no 'stuck' event). "
                "Exit nodes are shown in orange. Click again to remove."
            )
        else:
            self.status.set(f"Tool: {t}")

    def _on_scrub_start(self, e=None):
        self._scrubbing = True

    def _on_scrub_end(self, e=None):
        self._scrubbing = False
        # keep view_time when stopped; when running, snap back to live
        if self._running:
            self.view_time = None

    def _on_time_scrub(self, _val=None):
        if self._running:
            return
        t = float(self.time_var.get())
        # Timeline is for review only: if you scrub to the past, disable editing and playback.
        eps = 1e-6
        if (self._time_max - t) > 0.05:
            self._review_mode = True
        else:
            self._review_mode = False

        self.view_time = t
        # restore world/sim snapshot closest to this time (true time travel)
        with self._snap_lock:
            if not self._snaps:
                self._render()
                return
            # find nearest index by time
            idx = 0
            best = 1e18
            for i, s in enumerate(self._snaps):
                d = abs(float(s.get('t', 0.0)) - t)
                if d < best:
                    best = d
                    idx = i
            self._playhead = idx
            snap = self._snaps[idx]
        try:
            self.sim.restore(snap)
            self.world = self.sim.world
        except Exception:
            pass
        self._render()
    def _on_time_scrub_end(self, _evt=None):
        # finalize scrub selection; allow future edits from this point
        self._dirty_from_scrub = True
        try:
            self.status.set(f"Time set to {float(self.time_var.get()):.2f}s")
        except Exception:
            pass

    def _vehicle_states_at(self, t_query: float):
        """Return last known (x,y,heading) per vehicle at or before t_query."""
        cache = getattr(self, "_states_cache", None)
        if cache is not None and abs(cache.get("t", -1.0) - t_query) < 1e-6:
            return cache.get("states", {})

        states = {}
        for ev in self._all_events:
            if getattr(ev, "topic", "") != "world.vehicle_state":
                continue
            p = getattr(ev, "payload", None) or {}
            try:
                t = float(p.get("t", 0.0))
            except Exception:
                continue
            if t > t_query:
                break
            vid = p.get("vehicle_id")
            if not vid:
                continue
            states[vid] = (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("heading_rad", 0.0)),
            )

        self._states_cache = {"t": t_query, "states": states}
        return states

    def _can_edit(self):
        # Editing is disabled while simulation is running, or when viewing the past via the timeline.
        if self._running:
            return False
        return not self._review_mode


    def _mouse_world(self, sx, sy):
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 600
        return self.vp.screen_to_world(sx, sy, cw, ch)

    def _mark_dirty(self):
        # If we are currently viewing the past (playhead not at end), any edit forks the timeline.
        with self._snap_lock:
            if self._snaps and self._playhead >= 0 and self._playhead < len(self._snaps) - 1:
                self._dirty_from_scrub = True
        
    def _nearest_node(self, wx, wy, thresh=16.0):
        best = None
        best_d = 1e18
        for nid, n in self.world.nodes.items():
            d = dist((wx, wy), (n.x, n.y))
            if d < best_d:
                best_d = d
                best = nid
        if best is None or best_d > thresh:
            return None
        return best

    def _new_id(self, prefix: str) -> str:
        i = 1
        while True:
            cid = f"{prefix}{i}"
            if prefix == "veh" and cid not in self.world.vehicles:
                return cid
            if prefix == "gps" and cid not in self.world.gps:
                return cid
            if prefix == "cam" and cid not in self.world.cameras:
                return cid
            if prefix == "das" and cid not in self.world.das:
                return cid
            i += 1

    def _pick_lane(self, wx, wy):
        best = None
        best_d = 1e18
        best_s = 0.0
        for lid, lane in self.world.lanes.items():
            s, _, d = polyline_nearest_s(lane.polyline, (wx, wy))
            if d < best_d:
                best_d = d
                best = lid
                best_s = s
        if best is None or best_d > 30.0:
            return None
        return (best, best_s)

    def _pick_segment(self, wx, wy):
        pick = self._pick_lane(wx, wy)
        if pick is None:
            return None
        lid, _ = pick
        lane = self.world.lanes.get(lid)
        return lane.segment_id if lane else None

    def _hit_test(self, wx, wy):
        for nid, n in self.world.nodes.items():
            if dist((wx, wy), (n.x, n.y)) <= 9.0:
                return ("node", nid)

        for vid, v in self.world.vehicles.items():
            lane = self.world.lanes.get(v.lane_id)
            if not lane:
                continue
            px, py = point_at_s(lane.polyline, v.s)
            if dist((wx, wy), (px, py)) <= 10.0:
                return ("vehicle", vid)

        for gid, g in self.world.gps.items():
            if dist((wx, wy), (g.x, g.y)) <= 12.0:
                return ("gps", gid)

        for cid, c in self.world.cameras.items():
            if dist((wx, wy), (c.x, c.y)) <= 12.0:
                return ("camera", cid)

        for did, d in self.world.das.items():
            if d.segment_id and self._pick_segment(wx, wy) == d.segment_id:
                return ("das", did)

        seg_id = self._pick_segment(wx, wy)
        if seg_id:
            return ("segment", seg_id)

        return None

    # ---------------- Mouse ----------------
    def _on_left_down(self, e):
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        wx, wy = self._mouse_world(e.x, e.y)
        t = self.tool.get()

        # Traffic marking mode: click segments to toggle their traffic level.
        if self._traffic_mark_active and self._traffic_mark_level is not None:
            seg_id = self._pick_segment(wx, wy)
            if not seg_id:
                self.status.set("Traffic marking: click on a road segment.")
                return
            seg = self.world.segments.get(seg_id)
            if not seg:
                return
            cur = (getattr(seg, "traffic_level", "none") or "none").strip().lower()
            new = "none" if cur == self._traffic_mark_level else self._traffic_mark_level
            seg.traffic_level = new
            self._traffic_last_seg = seg_id
            self._traffic_last_ts = time.time()
            self.status.set(f"Traffic TAG MODE [{self._traffic_mark_level.upper()}]: {seg_id} → {new}. (Click again to untag; Done/Esc to exit)")
            self._render()
            return

        # Route planning click collection (independent of tool)
        if self.pending and self.pending.get("kind") == "plan_route":
            pick = self._pick_lane(wx, wy)
            if pick is None:
                self.status.set("Plan route: please click on a road.")
                return
            lane_id, s = pick
            self.pending["wps"].append((lane_id, float(s)))
            self.status.set(f"Plan route: added waypoint {len(self.pending['wps'])}. Press Enter to commit.")
            self._render()
            return

        # Select vehicle even when not in select tool (prevents accidental add-on-click)
        hit_any = self._hit_test(wx, wy)
        if hit_any and hit_any[0] == "vehicle" and t != TOOL_SELECT:
            self.selected = hit_any
            self._update_properties()
            self._render()
            return

        if t == TOOL_SELECT:
            hit = self._hit_test(wx, wy)
            self.selected = hit
            self._update_properties()
            if hit:
                # snapshot before drag for undo
                self._drag_world_before = self.undo.snapshot(self.world)
                self._drag_snapshot_taken = True

                if hit[0] == "node":
                    self.dragging_node = hit[1]
                elif hit[0] == "vehicle":
                    self.dragging_vehicle = hit[1]
                elif hit[0] in ("gps", "camera"):
                    self.dragging_sensor = (hit[0], hit[1])
            return

        # For add operations: push undo snapshot once per completed action
        if t == TOOL_SEGMENT:
            self._handle_add_segment(wx, wy)
            return
        if t == TOOL_VEHICLE:
            self._handle_add_vehicle(wx, wy)
            return
        if t == TOOL_ROUTE_VEHICLE:
            # legacy tool id; treat as routed vehicle add
            self.vehicle_add_mode.set("routed")
            self._handle_add_vehicle(wx, wy)
            return
        if t == TOOL_GPS:
            self._handle_add_gps(wx, wy)
            return
        if t == TOOL_CAMERA:
            self._handle_add_camera(wx, wy)
            return
        if t == TOOL_DAS:
            self._handle_add_das(wx, wy)
            return
        if t == TOOL_BOUNDARY_EXIT:
            self._handle_toggle_exit_node(wx, wy)
            return

    def _on_left_drag(self, e):
        wx, wy = self._mouse_world(e.x, e.y)

        if self.dragging_node:
            n = self.world.nodes.get(self.dragging_node)
            if n:
                n.x, n.y = wx, wy
                self.sim.rebuild_lanes()
                self._render()
                self._update_properties()

        if self.dragging_vehicle:
            v = self.world.vehicles.get(self.dragging_vehicle)
            if v:
                pick = self._pick_lane(wx, wy)
                if pick is None:
                    return
                lane_id, s = pick
                if lane_id:
                    v.lane_id = lane_id
                    v.s = s
                    self._render()
                    self._update_properties()

        if self.dragging_sensor:
            kind, sid = self.dragging_sensor
            if kind == "gps":
                g = self.world.gps.get(sid)
                if g:
                    g.x, g.y = wx, wy
                    self._render()
                    self._update_properties()
            if kind == "camera":
                c = self.world.cameras.get(sid)
                if c:
                    # Shift-drag rotates camera, normal drag moves it
                    if (e.state & 0x0001) != 0:
                        c.heading_rad = math.atan2(wy - c.y, wx - c.x)
                    else:
                        c.x, c.y = wx, wy
                    self._render()
                    self._update_properties()

        if self.tool.get() == TOOL_CAMERA and self.pending and self.pending.get("kind") == "camera_dir":
            cid = self.pending["cam_id"]
            c = self.world.cameras.get(cid)
            if c:
                c.heading_rad = math.atan2(wy - c.y, wx - c.x)
                self._render()

    def _on_left_up(self, e):
        dragged = (self.dragging_node is not None) or (self.dragging_vehicle is not None) or (self.dragging_sensor is not None)
        self.dragging_node = None
        self.dragging_vehicle = None
        self.dragging_sensor = None

        if dragged and self._drag_snapshot_taken:
            after = self.undo.snapshot(self.world)
            if self._drag_world_before and after != self._drag_world_before:
                self.undo.undo.append(self._drag_world_before)
                self.undo._last = after
                self.undo.redo = []
            self._drag_snapshot_taken = False
            self._drag_world_before = None
            self.status.set("Edit applied (drag).")

        if self.tool.get() == TOOL_CAMERA and self.pending and self.pending.get("kind") == "camera_dir":
            self.pending = None
            self.undo.push(self.world)
            self.status.set("Camera direction set.")

    def _on_pan_down(self, e):
        self.dragging_pan = (e.x, e.y)

    def _on_pan_drag(self, e):
        if not self.dragging_pan:
            return
        x0, y0 = self.dragging_pan
        dx = (e.x - x0) / self.vp.scale
        dy = (e.y - y0) / self.vp.scale
        self.vp.ox += dx
        self.vp.oy += dy
        self.dragging_pan = (e.x, e.y)
        self._render()

    def _on_pan_up(self, e):
        self.dragging_pan = None

    def _on_wheel(self, e):
        if e.delta > 0:
            self._zoom(1.1, e.x, e.y)
        else:
            self._zoom(0.9, e.x, e.y)

    def _zoom(self, factor, sx, sy):
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 600
        wx, wy = self.vp.screen_to_world(sx, sy, cw, ch)
        self.vp.scale *= factor
        self.vp.scale = max(0.25, min(6.0, self.vp.scale))
        self.vp.ox = (sx - cw / 2) / self.vp.scale - wx
        self.vp.oy = (sy - ch / 2) / self.vp.scale - wy
        self._render()

    # ---------------- Add handlers ----------------
    def _handle_add_segment(self, wx, wy):
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        self._mark_dirty()
        if self.pending is None:
            self.undo.push(self.world)
            nid = self._nearest_node(wx, wy)
            if nid is None:
                nid = f"n{len(self.world.nodes) + 1}"
                self.world.nodes[nid] = Node(id=nid, x=wx, y=wy)
            self.pending = {"kind": "segment", "n0": nid}
            self.status.set("Segment: click end point (snaps to existing nodes).")
            self._render()
            return

        if self.pending.get("kind") == "segment":
            nid = self._nearest_node(wx, wy)
            if nid is None:
                nid = f"n{len(self.world.nodes) + 1}"
                self.world.nodes[nid] = Node(id=nid, x=wx, y=wy)
            n0 = self.pending["n0"]
            sid = f"seg{len(self.world.segments) + 1}"
            self.segment_lanes.set(1)
            self.world.segments[sid] = Segment(id=sid, n0=n0, n1=nid, lanes=1, lane_width=3.6, one_way=True)
            self.pending = None
            self.sim.rebuild_lanes()
            self.undo.push(self.world)
            self.status.set(f"Created {sid}. Drag nodes to edit.")
            self._render()

    def _handle_add_vehicle(self, wx, wy):
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        self._mark_dirty()
        # Place a vehicle (free by default). Route planning is done from Properties (press 'p' or click the button).
        pick = self._pick_lane(wx, wy)
        if pick is None:
            messagebox.showerror("Place vehicle", "Click on a road (segment/lane) to place a vehicle.")
            return
        lane_id, s = pick
        vid = self._new_id("veh")
        seg = self.world.segments.get(self.world.lanes[lane_id].segment_id)
        profile = _random_vehicle_macro_profile(float(getattr(seg, "speed_limit_mps", 13.9) or 13.9))
        init_v = max(profile["speed_min_mps"], min(profile["speed_max_mps"], profile["target_speed_mps"]))
        self.world.vehicles[vid] = Vehicle(id=vid, lane_id=lane_id, s=float(s), v=float(init_v), weight_kg=1500.0, **profile)
        self.undo.push(self.world)
        self.status.set(f"Added {vid}. Select it to plan a route.")
        self._render()

    def _handle_add_gps(self, wx, wy):
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        self._mark_dirty()
        self.undo.push(self.world)
        gid = f"gps{len(self.world.gps) + 1}"
        self.world.gps[gid] = GPSSensor(id=gid, x=wx, y=wy)
        self.undo.push(self.world)
        self.status.set(f"Added {gid}.")
        self._render()

    def _handle_add_camera(self, wx, wy):
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        self._mark_dirty()
        self.undo.push(self.world)
        cid = f"cam{len(self.world.cameras) + 1}"
        self.world.cameras[cid] = CameraSensor(id=cid, x=wx, y=wy)
        self.pending = {"kind": "camera_dir", "cam_id": cid}
        self.status.set("Camera placed. Drag to set direction.")
        self._render()

    def _handle_add_das(self, wx, wy):
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        self._mark_dirty()
        self.undo.push(self.world)
        seg_id = self._pick_segment(wx, wy)
        if seg_id is None:
            self.status.set("DAS must be attached to a segment (click on a segment).")
            return
        did = f"das{len(self.world.das) + 1}"
        self.world.das[did] = DASSensor(id=did, segment_id=seg_id)
        self.undo.push(self.world)
        self.status.set(f"Added {did} on {seg_id}.")
        self._render()

    def _handle_toggle_exit_node(self, wx, wy):
        """Toggle the nearest node in/out of world.boundary_exit_nodes."""
        if not self._can_edit():
            self.status.set('Timeline is in review mode — editing disabled.')
            return
        # Snap to the nearest node within a generous click radius (20 m world units).
        nid = self._nearest_node(wx, wy, thresh=20.0)
        if nid is None:
            self.status.set("Exit Node: no node found nearby — click closer to a node.")
            return
        self._mark_dirty()
        self.undo.push(self.world)
        exits = getattr(self.world, "boundary_exit_nodes", None)
        if exits is None:
            self.world.boundary_exit_nodes = set()
            exits = self.world.boundary_exit_nodes
        if nid in exits:
            exits.discard(nid)
            self.status.set(f"Exit Node: {nid} removed from route endpoints.")
        else:
            exits.add(nid)
            self.status.set(
                f"Exit Node: {nid} marked as route endpoint. "
                "Vehicles that reach this node exit the map cleanly."
            )
        self._render()

    # ---------------- Delete ----------------
    def _delete_selection(self):
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry)):
            w.delete(0, 'end')
            return
        if isinstance(w, tk.Text):
            try:
                w.delete('sel.first', 'sel.last')
            except Exception:
                pass
            return
        self._mark_dirty()
        if not self.selected:
            return
        self.undo.push(self.world)

        kind, sid = self.selected
        if kind == "node":
            seg_to_del = [k for k, s in self.world.segments.items() if s.n0 == sid or s.n1 == sid]
            for k in seg_to_del:
                self.world.segments.pop(k, None)
            self.world.nodes.pop(sid, None)
            for did in list(self.world.das.keys()):
                if self.world.das[did].segment_id in seg_to_del:
                    self.world.das.pop(did, None)
            self.sim.rebuild_lanes()

        elif kind == "segment":
            self.world.segments.pop(sid, None)
            for did in list(self.world.das.keys()):
                if self.world.das[did].segment_id == sid:
                    self.world.das.pop(did, None)
            self.sim.rebuild_lanes()

        elif kind == "vehicle":
            self.world.vehicles.pop(sid, None)
        elif kind == "gps":
            self.world.gps.pop(sid, None)
        elif kind == "camera":
            self.world.cameras.pop(sid, None)
        elif kind == "das":
            self.world.das.pop(sid, None)

        self.selected = None
        self.undo.push(self.world)
        self._update_properties()
        self._render()
        self.status.set("Deleted selection.")

    # ---------------- Route planning ----------------
    def _begin_plan_route(self):
        if not self.selected or self.selected[0] != "vehicle":
            return
        vid = self.selected[1]
        if vid not in self.world.vehicles:
            return
        # Route planning state-machine:
        # - pending['wps'] collects (lane_id, s) points (snapped to lanes)
        # - Commit is done by the in-UI "Apply Route" button (and also Enter).
        self.pending = {"kind": "plan_route", "veh_id": vid, "wps": []}
        self.status.set("Plan route: click points on roads. Then press Apply Route (or Enter). Esc cancels.")
        self._render()

    def _clear_selected_route(self):
        if not self.selected or self.selected[0] != "vehicle":
            return
        vid = self.selected[1]
        self.sim.clear_route(vid)
        self._vehicle_routes.pop(str(vid), None)
        self._route_progress.pop(str(vid), None)
        self.status.set(f"Vehicle {vid}: route cleared.")
        self._render()

    def _commit_pending(self):
        if not self.pending:
            return
        if self.pending.get("kind") != "plan_route":
            return
        vid = self.pending.get("veh_id")
        wps = list(self.pending.get("wps") or [])
        self.pending = None
        if not wps:
            self.status.set("Plan route: no waypoints selected.")
            self._render()
            return
        self.sim.rebuild_lanes()
        ok = self.sim.set_route_queue(vid, wps)
        if not ok:
            messagebox.showerror("No route", "Cannot reach the first waypoint from the vehicle's current position. Choose different points.")
            return
        self._vehicle_routes[str(vid)] = list(wps)
        self._route_progress[str(vid)] = 0
        self.undo.push(self.world)
        self.status.set(f"Vehicle {vid}: route set with {len(wps)} waypoint(s).")
        self.pending = None  # exit route planning mode after commit
        self._render()

    def _cancel_pending(self):
        if not self.pending:
            return
        if self.pending.get("kind") == "plan_route":
            self.pending = None
            self.status.set("Plan route cancelled.")
            self._render()

    # ---------------- Properties ----------------
    def _clear_prop_frame(self):
        for w in list(self.prop_frame.winfo_children()):
            if w is self.apply_btn:
                continue
            w.destroy()
        self.prop_entries = {}
        self.prop_vars = {}
        self._prop_row = 0

    def _add_entry(self, key, label, value):
        row = self._prop_row
        ttk.Label(self.prop_frame, text=label).grid(row=row, column=0, sticky="w")
        e = ttk.Entry(self.prop_frame)
        e.grid(row=row, column=1, sticky="ew")
        e.insert(0, str(value))
        self.prop_entries[key] = e
        self._prop_row += 1

    def _add_slider(self, key, label, value, min_value, max_value, digits=1):
        row = self._prop_row
        ttk.Label(self.prop_frame, text=label).grid(row=row, column=0, sticky="w")
        box = ttk.Frame(self.prop_frame)
        box.grid(row=row, column=1, sticky="ew")
        box.columnconfigure(0, weight=1)
        var = tk.DoubleVar(value=float(value))
        fmt = '{:.' + str(int(digits)) + 'f}'
        value_lbl = ttk.Label(box, width=max(6, digits + 4), text=fmt.format(float(value)))
        def _on_change(*_):
            try:
                value_lbl.configure(text=fmt.format(float(var.get())))
            except Exception:
                pass
        var.trace_add('write', _on_change)
        sc = ttk.Scale(box, from_=float(min_value), to=float(max_value), variable=var)
        sc.grid(row=0, column=0, sticky='ew', padx=(0, 6))
        value_lbl.grid(row=0, column=1, sticky='e')
        self.prop_vars[key] = var
        self._prop_row += 1

    def _sync_vehicle_slider_ranges(self):
        keys = {"speed_min_kmh", "speed_mean_kmh", "speed_max_kmh", "target_speed_kmh"}
        if not keys.intersection(self.prop_vars):
            return
        try:
            speed_min = float(self.prop_vars.get("speed_min_kmh").get()) if self.prop_vars.get("speed_min_kmh") is not None else 0.0
            speed_mean = float(self.prop_vars.get("speed_mean_kmh").get()) if self.prop_vars.get("speed_mean_kmh") is not None else speed_min
            speed_max = float(self.prop_vars.get("speed_max_kmh").get()) if self.prop_vars.get("speed_max_kmh") is not None else max(speed_mean + 5.0, speed_min + 5.0)
            speed_max = max(speed_max, speed_min + 5.0)
            speed_mean = min(max(speed_mean, speed_min), speed_max)
            if self.prop_vars.get("speed_max_kmh") is not None and float(self.prop_vars["speed_max_kmh"].get()) != speed_max:
                self.prop_vars["speed_max_kmh"].set(speed_max)
            if self.prop_vars.get("speed_mean_kmh") is not None and float(self.prop_vars["speed_mean_kmh"].get()) != speed_mean:
                self.prop_vars["speed_mean_kmh"].set(speed_mean)
            if self.prop_vars.get("target_speed_kmh") is not None:
                target = float(self.prop_vars["target_speed_kmh"].get())
                target = min(max(target, speed_min), speed_max)
                if float(self.prop_vars["target_speed_kmh"].get()) != target:
                    self.prop_vars["target_speed_kmh"].set(target)
        except Exception:
            pass

    def _apply_properties(self):
        # If we're in route-planning mode, the main "Apply" button acts as
        # "Apply Route" for the currently selected vehicle.
        if self.pending and self.pending.get("kind") == "plan_route":
            if self.selected and self.selected[0] == "vehicle" and self.selected[1] == self.pending.get("veh_id"):
                self._commit_pending()
                return
        if not self.selected:
            return

        def get_float(key, default):
            ent = self.prop_entries.get(key)
            if ent is not None:
                try:
                    return float(ent.get())
                except Exception:
                    return default
            var = self.prop_vars.get(key)
            if var is not None:
                try:
                    return float(var.get())
                except Exception:
                    return default
            return default

        kind, sid = self.selected

        if kind == "vehicle":
            v = self.world.vehicles.get(sid)
            if v:
                # UI is in km/h; engine stores m/s
                v_kmh = get_float("v_kmh", v.v * 3.6)
                v.v = max(0.0, float(v_kmh) / 3.6)
                v.weight_kg = max(500.0, get_float("weight_kg", v.weight_kg))
                speed_min = max(0.0, get_float("speed_min_kmh", max(0.0, getattr(v, "speed_min_mps", v.v) * 3.6))) / 3.6
                speed_max = max(0.0, get_float("speed_max_kmh", max(speed_min * 3.6, getattr(v, "speed_max_mps", v.v) * 3.6))) / 3.6
                if speed_max <= speed_min:
                    speed_max = speed_min + (5.0 / 3.6)
                speed_mean = max(speed_min, min(speed_max, get_float("speed_mean_kmh", max(speed_min * 3.6, min(speed_max * 3.6, getattr(v, "speed_mean_mps", v.v) * 3.6))) / 3.6))
                target_speed = max(speed_min, min(speed_max, get_float("target_speed_kmh", max(speed_min * 3.6, min(speed_max * 3.6, getattr(v, "target_speed_mps", v.v) * 3.6))) / 3.6))
                interval_min = max(1.0, get_float("speed_change_interval_min_s", max(1.0, getattr(v, "speed_change_interval_min_s", 4.0))))
                interval_max = max(interval_min + 0.5, get_float("speed_change_interval_max_s", max(interval_min + 0.5, getattr(v, "speed_change_interval_max_s", 10.0))))
                interval_mean = 0.5 * (interval_min + interval_max)
                max_accel_mps2 = max(0.2, get_float("max_accel_mps2", max(0.2, getattr(v, "max_accel_mps2", 1.8))))
                max_decel_mps2 = max(0.2, get_float("max_decel_mps2", max(0.2, getattr(v, "max_decel_mps2", 2.6))))
                speed_std = max(0.5, 0.18 * max(1.0, speed_max - speed_min))
                v.speed_min_mps = speed_min
                v.speed_max_mps = speed_max
                v.speed_mean_mps = speed_mean
                v.speed_std_mps = speed_std
                v.target_speed_mps = target_speed
                v.speed_change_interval_mean_s = interval_mean
                v.speed_change_interval_min_s = interval_min
                v.speed_change_interval_max_s = interval_max
                v.next_speed_change_t = 0.0
                v.last_speed_change_t = -1e9
                v.cruise_hold_probability = 0.0
                v.accel_response_s = 1.5
                v.max_accel_mps2 = max_accel_mps2
                v.max_decel_mps2 = max_decel_mps2
                try:
                    self.sim.apply_vehicle_macro_profile(str(sid))
                except Exception:
                    pass
                # Vehicles are modeled as points; ignore any legacy footprint fields.

        if kind == "gps":
            g = self.world.gps.get(sid)
            if g:
                g.sigma_m = get_float("sigma_m", g.sigma_m)
                g.radius_m = get_float("radius_m", g.radius_m)
                g.update_hz = get_float("update_hz", g.update_hz)

        if kind == "camera":
            c = self.world.cameras.get(sid)
            if c:
                c.fov_deg = get_float("fov_deg", c.fov_deg)
                c.range_m = get_float("range_m", c.range_m)
                c.update_hz = get_float("update_hz", c.update_hz)

        if kind == "das":
            d = self.world.das.get(sid)
            if d:
                d.channel_start = int(get_float("channel_start", d.channel_start))
                d.channel_end = int(get_float("channel_end", d.channel_end))
                d.update_hz = get_float("update_hz", d.update_hz)
                d.noise_std = get_float("noise_std", d.noise_std)

        self.sim.rebuild_lanes()
        self.undo.push(self.world)
        self._update_properties()
        self._render()
        self.status.set("Properties applied.")

    def _update_properties(self):
        self.props.configure(state="normal")
        self.props.delete("1.0", "end")
        self._clear_prop_frame()

        # Default Apply label; in route-planning mode this becomes "Apply Route".
        try:
            self.apply_btn.configure(text="Apply")
        except Exception:
            pass

        if not self.selected:
            self.props.insert("end", "No selection.\n\nShortcuts:\n- V: select/move\n- S: segment\n- R: vehicle\n- G: GPS\n- C: camera\n- D: DAS\n- Cmd/Ctrl+Z: undo")
            self.props.configure(state="disabled")
            return

        kind, sid = self.selected

        if kind == "vehicle":
            v = self.world.vehicles.get(sid)
            if v:
                self.props.insert(
                    "end",
                    f"Vehicle {sid}\n lane={v.lane_id}\n s={v.s:.2f}\n v={v.v * 3.6:.1f} km/h\n weight_kg={v.weight_kg:.1f}",
                )
                self._add_slider("v_kmh", "Current speed (km/h)", v.v * 3.6, 0.0, max(120.0, max(v.v * 3.6 + 10.0, getattr(v, "speed_max_mps", v.v) * 3.6 + 10.0)), digits=1)
                self._add_entry("weight_kg", "weight_kg", v.weight_kg)
                speed_min_kmh = max(0.0, float(getattr(v, "speed_min_mps", max(0.0, v.v - 2.0)) or max(0.0, v.v - 2.0)) * 3.6)
                speed_max_kmh = max(speed_min_kmh + 5.0, float(getattr(v, "speed_max_mps", max(v.v + 2.0, 6.0)) or max(v.v + 2.0, 6.0)) * 3.6)
                speed_mean_kmh = max(speed_min_kmh, min(speed_max_kmh, float(getattr(v, "speed_mean_mps", v.v) or v.v) * 3.6))
                interval_min_s = max(1.0, float(getattr(v, "speed_change_interval_min_s", 4.0) or 4.0))
                interval_max_s = max(interval_min_s + 0.5, float(getattr(v, "speed_change_interval_max_s", 10.0) or 10.0))
                self._add_slider("speed_min_kmh", "Min speed (km/h)", speed_min_kmh, 0.0, 120.0, digits=1)
                self._add_slider("speed_mean_kmh", "Avg speed (km/h)", speed_mean_kmh, 0.0, 120.0, digits=1)
                self._add_slider("speed_max_kmh", "Max speed (km/h)", speed_max_kmh, 5.0, 140.0, digits=1)
                self._add_slider("max_accel_mps2", "Max accel (m/s²)", max(0.2, float(getattr(v, "max_accel_mps2", 1.8) or 1.8)), 0.2, 5.0, digits=2)
                self._add_slider("max_decel_mps2", "Max brake (m/s²)", max(0.2, float(getattr(v, "max_decel_mps2", 2.6) or 2.6)), 0.2, 6.0, digits=2)
                self._add_slider("speed_change_interval_min_s", "Min change interval (s)", interval_min_s, 1.0, 20.0, digits=1)
                self._add_slider("speed_change_interval_max_s", "Max change interval (s)", interval_max_s, 2.0, 30.0, digits=1)
                for _slider_key in ("speed_min_kmh", "speed_mean_kmh", "speed_max_kmh"):
                    try:
                        self.prop_vars[_slider_key].trace_add("write", lambda *_args: self._sync_vehicle_slider_ranges())
                    except Exception:
                        pass
                ttk.Button(self.prop_frame, text="Randomize realistic profile", command=self._randomize_selected_vehicle_profile).grid(row=self._prop_row, column=0, columnspan=2, sticky="ew", pady=(6, 2))
                self._prop_row += 1
                # Vehicles are modeled as points; no footprint editing.
                self.props.insert('end', '\n\nRoute waypoints (shown on map only when this vehicle is selected):\n')
                q = self._vehicle_routes.get(str(sid)) or []
                prog = int(self._route_progress.get(str(sid), 0))
                self.props.tag_configure('reached', foreground='#ff3b3b')
                if not q:
                    self.props.insert('end', '  (none)\n')
                else:
                    for i, (lane_id, s) in enumerate(q, 1):
                        txt = f'  {i}. lane={lane_id}  s={float(s):.2f}\n'
                        if (i - 1) < prog:
                            self.props.insert('end', txt, ('reached',))
                        else:
                            self.props.insert('end', txt)
                # Routing helpers
                ttk.Separator(self.prop_frame).grid(row=50, column=0, columnspan=2, sticky="ew", pady=(10, 6))
                ttk.Button(self.prop_frame, text="Plan Route… (P)", command=self._begin_plan_route).grid(row=51, column=0, columnspan=2, sticky="ew", pady=2)
                ttk.Button(self.prop_frame, text="Clear Route", command=self._clear_selected_route).grid(row=52, column=0, columnspan=2, sticky="ew", pady=2)

                # Planning mode UX: we use the main Apply button as "Apply Route"
                # so users do not have to memorize the Enter shortcut.
                if self.pending and self.pending.get("kind") == "plan_route" and self.pending.get("veh_id") == sid:
                    self.apply_btn.configure(text="Apply Route")
                    ttk.Button(self.prop_frame, text="Cancel Route", command=self._cancel_pending).grid(row=53, column=0, columnspan=2, sticky="ew", pady=2)
                    self.props.insert('end', '\n\nPlanning mode:\n  - Click points on roads to add waypoints\n  - Press Apply Route to start immediately\n  - Esc / Cancel to abort\n')

        elif kind == "gps":
            g = self.world.gps.get(sid)
            if g:
                self.props.insert("end", f"GPS {sid}\n x={g.x:.2f}\n y={g.y:.2f}\n sigma_m={g.sigma_m}\n radius_m={g.radius_m}\n update_hz={g.update_hz}")
                self._add_entry("sigma_m", "sigma_m", g.sigma_m)
                self._add_entry("radius_m", "radius_m", g.radius_m)
                self._add_entry("update_hz", "update_hz", g.update_hz)

        elif kind == "camera":
            c = self.world.cameras.get(sid)
            if c:
                self.props.insert("end", f"Camera {sid}\n x={c.x:.2f}\n y={c.y:.2f}\n fov_deg={c.fov_deg}\n range_m={c.range_m}\n update_hz={c.update_hz}")
                self._add_entry("fov_deg", "fov_deg", c.fov_deg)
                self._add_entry("range_m", "range_m", c.range_m)
                self._add_entry("update_hz", "update_hz", c.update_hz)

        elif kind == "das":
            d = self.world.das.get(sid)
            if d:
                self.props.insert("end", f"DAS {sid}\n segment_id={d.segment_id}\n channel_start={d.channel_start}\n channel_end={d.channel_end}\n update_hz={d.update_hz}\n noise_std={d.noise_std}")
                self._add_entry("channel_start", "channel_start", d.channel_start)
                self._add_entry("channel_end", "channel_end", d.channel_end)
                self._add_entry("update_hz", "update_hz", d.update_hz)
                self._add_entry("noise_std", "noise_std", d.noise_std)

        elif kind == "node":
            n = self.world.nodes.get(sid)
            if n:
                self.props.insert("end", f"Node {sid}\n x={n.x:.2f}\n y={n.y:.2f}")

        elif kind == "segment":
            s = self.world.segments.get(sid)
            if s:
                typ = "one_way" if getattr(s, "one_way", False) else "two_way"
                self.props.insert(
                    "end",
                    f"Segment {sid}\n n0={s.n0}\n n1={s.n1}\n type={typ}\n lanes_per_direction={s.lanes}\n lane_width={s.lane_width}",
                )

        self.apply_btn.grid(row=max(self._prop_row + 1, 99), column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.props.configure(state="disabled")

    # ---------------- Rendering ----------------

    # ---------------- Drawing helpers ----------------
    def _draw_arrows_on_poly(self, poly, cw, ch, forward=True, color="#c7cdd6"):
        """Draw repeated directional arrows along a world-space polyline."""
        try:
            Lw = float(polyline_length(poly))
        except Exception:
            return
        if Lw <= 1e-6:
            return

        # Keep arrow spacing roughly constant in screen space.
        spacing_px = 70.0
        spacing_w = spacing_px / max(1e-6, float(self.vp.scale))
        delta_w = 2.0 / max(1e-6, float(self.vp.scale))  # for heading estimation

        # Arrow size in pixels
        size = max(6.0, 10.0 * float(self.vp.scale))
        half_w = size * 0.55
        len_h = size * 1.1

        s = spacing_w * 0.5
        while s < (Lw - spacing_w * 0.5):
            p = point_at_s(poly, s)
            p2 = point_at_s(poly, min(Lw, s + delta_w))
            ang = math.atan2(p2[1] - p[1], p2[0] - p[0])
            if not forward:
                ang += math.pi

            sx, sy = self.vp.world_to_screen(p[0], p[1], cw, ch)
            c = math.cos(ang)
            sn = math.sin(ang)

            # Triangle points in screen space
            tip = (sx + c * len_h, sy + sn * len_h)
            left = (sx - c * len_h * 0.25 - sn * half_w, sy - sn * len_h * 0.25 + c * half_w)
            right = (sx - c * len_h * 0.25 + sn * half_w, sy - sn * len_h * 0.25 - c * half_w)

            self.canvas.create_polygon(
                tip[0], tip[1], left[0], left[1], right[0], right[1],
                fill=color, outline=""
            )

            s += spacing_w

    def _render(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 600

        # background grid
        for x in range(0, cw, 80):
            self.canvas.create_line(x, 0, x, ch, fill="#141821")
        for y in range(0, ch, 80):
            self.canvas.create_line(0, y, cw, y, fill="#141821")

        # Roads
        # Render per segment, then per lane.
        # Visual rules:
        # - One-way: thinner road + arrows along the lane direction.
        # - Two-way: thicker road + two arrow streams (right = forward, left = backward).
        for seg_id, seg in self.world.segments.items():
            n0 = self.world.nodes.get(seg.n0)
            n1 = self.world.nodes.get(seg.n1)
            if not n0 or not n1:
                continue

            # Base road polyline (curves supported via Segment.points).
            base_poly = list(getattr(seg, 'points', []) or [(n0.x, n0.y), (n1.x, n1.y)])
            base_pts = []
            for (x, y) in base_poly:
                sx, sy = self.vp.world_to_screen(x, y, cw, ch)
                base_pts += [sx, sy]
            if len(base_pts) < 4:
                continue

            # Default matches the Segment dataclass default (one_way=True).
            is_one_way = bool(getattr(seg, "one_way", True))

            # Base road body (asphalt)
            if is_one_way:
                base_w = max(6, int(10 * self.vp.scale))
            else:
                base_w = max(8, int(16 * self.vp.scale))
            self.canvas.create_line(*base_pts, fill="#1e232b", width=base_w, capstyle="round", smooth=True)

            # Traffic overlay is shown only while tagging segments (traffic marking mode).
            # After the user clicks "Done" we keep the data (seg.traffic_level) but hide
            # the overlay to reduce visual clutter.
            if self._traffic_mark_active and self._traffic_mark_level is not None:
                lvl = (getattr(seg, "traffic_level", "none") or "none").strip().lower()
                if lvl != "none":
                    col = {"light": "#2aa745", "medium": "#d7b600", "heavy": "#d65c2a"}.get(lvl, "#d7b600")
                    is_last = (self._traffic_last_seg == seg.id) and ((time.time() - float(getattr(self, "_traffic_last_ts", 0.0))) < 0.8)
                    if is_last:
                        self.canvas.create_line(
                            *base_pts,
                            fill="#ffffff",
                            width=max(6, int(12 * self.vp.scale)),
                            capstyle="round",
                            smooth=True,
                        )
                    self.canvas.create_line(
                        *base_pts,
                        fill=col,
                        width=max(4, int(7 * self.vp.scale)),
                        capstyle="round",
                        smooth=True,
                    )
                    # Label at segment midpoint for quick confirmation (L/M/H)
                    mid = base_poly[len(base_poly) // 2]
                    msx, msy = self.vp.world_to_screen(mid[0], mid[1], cw, ch)
                    tag = {"light": "L", "medium": "M", "heavy": "H"}.get(lvl, "?")
                    self.canvas.create_text(msx, msy - 10, text=tag, fill=col, font=("Helvetica", 10, "bold"))

            # Lanes (already offset geometrically)
            for lid, lane in self.world.lanes.items():
                if lane.segment_id != seg_id:
                    continue
                pts = []
                for (x, y) in lane.polyline:
                    sx, sy = self.vp.world_to_screen(x, y, cw, ch)
                    pts += [sx, sy]
                if len(pts) < 4:
                    continue

                is_bwd = "_bwd_" in lid
                lane_fill = "#3a404a" if not is_bwd else "#2f353e"
                edge_fill = "#505866" if not is_bwd else "#454c58"

                if is_one_way:
                    w_lane = max(4, int(6 * self.vp.scale))
                    w_edge = max(2, int(2.5 * self.vp.scale))
                else:
                    w_lane = max(5, int(8 * self.vp.scale))
                    w_edge = max(2, int(3 * self.vp.scale))

                # lane body + edge (gives separation feel)
                self.canvas.create_line(*pts, fill=lane_fill, width=w_lane, capstyle="round")
                self.canvas.create_line(*pts, fill=edge_fill, width=w_edge, capstyle="round")

                # Direction arrows
                if is_one_way:
                    # Only forward lanes exist in the world for one-way segments, but keep robust.
                    self._draw_arrows_on_poly(lane.polyline, cw, ch, forward=True, color="#7b8491")
                else:
                    # Two-way: show arrows on both directions (right=forward, left=backward)
                    self._draw_arrows_on_poly(lane.polyline, cw, ch, forward=True, color="#c7cdd6")
# nodes
        _exit_nodes = getattr(self.world, "boundary_exit_nodes", set()) or set()
        for nid, n in self.world.nodes.items():
            sx, sy = self.vp.world_to_screen(n.x, n.y, cw, ch)
            is_exit = nid in _exit_nodes
            # Open-end warning ring (red) — nodes with no outgoing roads.
            if nid in getattr(self, "_open_end_nodes", set()):
                self.canvas.create_oval(sx - 11, sy - 11, sx + 11, sy + 11, outline="#ff3b3b", width=2)
            # Boundary-exit node: orange filled dot + thick orange outer ring.
            if is_exit:
                self.canvas.create_oval(sx - 13, sy - 13, sx + 13, sy + 13, outline="#FF8C00", width=3)
                self.canvas.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, fill="#FF8C00", outline="")
            else:
                self.canvas.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, fill="#d0d4da", outline="")
            # Node label — orange text for exit nodes so it stands out.
            lbl_col = "#FF8C00" if is_exit else "#aeb4bf"
            self.canvas.create_text(sx + 12, sy - 12, text=nid, fill=lbl_col, anchor="w", font=("Helvetica", 9))

        # GPS (radius + marker)
        for gid, g in self.world.gps.items():
            sx, sy = self.vp.world_to_screen(g.x, g.y, cw, ch)
            r = g.radius_m * self.vp.scale
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, outline="#35c9ff", width=2)
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#35c9ff", outline="")
            self.canvas.create_text(sx + 10, sy, text=gid, fill="#35c9ff", anchor="w", font=("Helvetica", 9))
            if self.selected == ("gps", gid):
                self.canvas.create_oval(sx - 12, sy - 12, sx + 12, sy + 12, outline="#ffffff", width=2)

        # Camera FOV (outline only; no fill)
        for cid, c in self.world.cameras.items():
            sx, sy = self.vp.world_to_screen(c.x, c.y, cw, ch)
            self.canvas.create_rectangle(sx - 6, sy - 6, sx + 6, sy + 6, fill="#7CFF6B", outline="")
            wedge = fov_wedge((c.x, c.y), c.heading_rad, math.radians(c.fov_deg), c.range_m)
            pts = []
            for (x, y) in wedge:
                px, py = self.vp.world_to_screen(x, y, cw, ch)
                pts += [px, py]
            if len(pts) >= 6:
                self.canvas.create_polygon(*pts, fill="", outline="#7CFF6B", width=3)
            self.canvas.create_text(sx + 10, sy, text=cid, fill="#7CFF6B", anchor="w", font=("Helvetica", 9))
            if self.selected == ("camera", cid):
                self.canvas.create_oval(sx - 14, sy - 14, sx + 14, sy + 14, outline="#ffffff", width=2)

        # DAS highlight
        for did, d in self.world.das.items():
            for lid, lane in self.world.lanes.items():
                if lane.segment_id != d.segment_id:
                    continue
                pts = []
                for (x, y) in lane.polyline:
                    sx, sy = self.vp.world_to_screen(x, y, cw, ch)
                    pts += [sx, sy]
                if len(pts) >= 4:
                    self.canvas.create_line(*pts, fill="#ffb020", width=3)

        # Vehicles are modeled as points. We visualize them as small discs where
        # fill darkness encodes weight (heavier => darker).
        #
        # Anomaly-aware status overlays.  Build sets of vehicle-ids that:
        #   * have been involved in a ``world.collision``  → red warning triangle
        #   * have published ``world.route_complete``      → blue X
        # Obstacles (Vehicle.object_type != "vehicle") are drawn as a yellow
        # warning diamond *instead* of the rectangle, regardless of events.
        # Use the persistent sets maintained by _ui_pump so markers stay visible
        # for the entire simulation run (not just while events are in the
        # rolling _all_events window).
        _collided_ids = self._persistent_collided_ids
        _route_complete_ids = self._persistent_route_complete_ids

        states = None
        if self.view_time is not None and not self._running:
            states = self._vehicle_states_at(float(self.view_time))
        for vid, v in self.world.vehicles.items():
            # When scrubbing, draw last known pose from dataset
            if states is not None and vid in states:
                x, y, ang = states[vid]
            else:
                lane = self.world.lanes.get(v.lane_id)
                if not lane:
                    continue
                x, y = point_at_s(lane.polyline, v.s)
                ang = float(getattr(v, 'heading_rad', 0.0))
                # heading_rad is 0.0 by default (dataclass) and only set meaningfully
                # once the simulation has run at least one tick.  Before that, derive
                # the angle from the lane-tangent so vehicles face the right direction
                # even before pressing Start.
                if not self._ever_played:
                    p_a = point_at_s(lane.polyline, max(0.0, v.s - 0.5))
                    p_b = point_at_s(lane.polyline, v.s + 0.5)
                    ang = math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
            sx, sy = self.vp.world_to_screen(x, y, cw, ch)
            # Vehicle appearance (point-based):
            # - Fill darkness encodes weight (log-mapped for better visual separation).
            wkg = float(getattr(v, "weight_kg", 1500.0) or 1500.0)

            # Weight -> color interpolation (light blue to dark blue)
            def _hex(r, g, b):
                return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

            def _lerp(a, b, t):
                return a + (b - a) * t

            # Use a wide realistic-ish range and a log mapping so that passenger cars
            # don't all look "the same" next to trucks/buses.
            wmin, wmax = 800.0, 36000.0
            lw = math.log(max(wmin, min(wkg, wmax)))
            t = (lw - math.log(wmin)) / max(1e-6, (math.log(wmax) - math.log(wmin)))
            t = max(0.0, min(1.0, t))
            light = (125, 190, 255)
            dark = (18, 46, 110)
            fill = _hex(_lerp(light[0], dark[0], t), _lerp(light[1], dark[1], t), _lerp(light[2], dark[2], t))

            # Size scaling with weight for readability (still point-based), but render as a small
            # *rectangle* (not a circle), rotated by heading.
            r = 4.0 + 5.0 * t
            half_len = r * 2.2
            half_wid = r * 1.2

            ca = math.cos(ang)
            sa = math.sin(ang)

            # Local -> screen rotation
            def _rot(dx, dy):
                return (sx + dx * ca - dy * sa, sy + dx * sa + dy * ca)

            # ---- Anomaly-aware base appearance --------------------------------
            # Road obstacles (Vehicle.object_type != "vehicle") are drawn as a
            # yellow road-safety warning sign instead of the regular vehicle
            # rectangle so they're immediately recognisable as obstacles.
            obj_type = str(getattr(v, "object_type", "vehicle") or "vehicle").lower()
            is_obstacle = (obj_type != "vehicle")

            if is_obstacle:
                # Yellow warning diamond (highway hazard sign), with a black
                # border + a "!" glyph so it reads as an obstacle marker.
                d = max(10.0, r * 2.6)  # half-diagonal in pixels
                pts_d = (sx, sy - d, sx + d, sy, sx, sy + d, sx - d, sy)
                self.canvas.create_polygon(*pts_d, fill="#ffd400", outline="#1a1a1a", width=2)
                # Exclamation mark inside.
                self.canvas.create_text(
                    sx, sy - 1, text="!", fill="#1a1a1a",
                    font=("Helvetica", max(9, int(d * 0.9)), "bold"),
                )
                # Label: "<id> (obj_type)" so the user can tell pedestrian /
                # debris / animal / barrier apart at a glance.
                self.canvas.create_text(
                    sx + d + 4, sy, text=f"{vid} ({obj_type})",
                    fill="#ffd400", anchor="w", font=("Helvetica", 9, "bold"),
                )
            else:
                p1 = _rot(+half_len, +half_wid)
                p2 = _rot(+half_len, -half_wid)
                p3 = _rot(-half_len, -half_wid)
                p4 = _rot(-half_len, +half_wid)
                self.canvas.create_polygon(
                    p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1],
                    fill=fill, outline=""
                )
                self.canvas.create_text(sx + 12, sy, text=vid, fill=fill, anchor="w", font=("Helvetica", 9))

            # ---- Selection highlight (works for both shapes) -----------------
            if self.selected == ("vehicle", vid):
                self.canvas.create_rectangle(sx - 16, sy - 16, sx + 16, sy + 16, outline="#ffffff", width=2)

            # ---- Anomaly status overlays --------------------------------------
            # Red warning triangle on every vehicle involved in a collision.
            # The same colour + same shape on both vehicles makes it obvious
            # that they are part of the same accident.
            if str(vid) in _collided_ids:
                tr = max(10.0, r * 2.4)
                p_top = (sx, sy - tr)
                p_bl = (sx - tr * 0.9, sy + tr * 0.7)
                p_br = (sx + tr * 0.9, sy + tr * 0.7)
                self.canvas.create_polygon(
                    p_top[0], p_top[1], p_br[0], p_br[1], p_bl[0], p_bl[1],
                    fill="#ff2a2a", outline="#1a1a1a", width=2,
                )
                self.canvas.create_text(
                    sx, sy + 2, text="!", fill="#ffffff",
                    font=("Helvetica", max(9, int(tr * 0.9)), "bold"),
                )

            # Blue X on every vehicle that has finished its planned route.
            # Distinct from the red X used for ``world.vehicle_stuck`` so the
            # user can tell "completed normally" from "dead-ended / crashed".
            if str(vid) in _route_complete_ids:
                xr = max(8.0, r * 1.8)
                self.canvas.create_line(
                    sx - xr, sy - xr, sx + xr, sy + xr,
                    fill="#3aa0ff", width=3,
                )
                self.canvas.create_line(
                    sx - xr, sy + xr, sx + xr, sy - xr,
                    fill="#3aa0ff", width=3,
                )

        # route planning waypoints (preview)
        if self.pending and self.pending.get("kind") == "plan_route":
            for i, (lane_id, s) in enumerate(self.pending.get("wps") or []):
                lane = self.world.lanes.get(lane_id)
                if not lane:
                    continue
                x, y = point_at_s(lane.polyline, float(s))
                sx, sy = self.vp.world_to_screen(x, y, cw, ch)
                self.canvas.create_oval(sx - 8, sy - 8, sx + 8, sy + 8, outline="#ffffff", width=2)
                self.canvas.create_text(sx + 12, sy, text=str(i + 1), fill="#ffffff", anchor="w", font=("Helvetica", 9, "bold"))

        # committed route waypoints (only shown for selected vehicle)
        if self.selected and self.selected[0] == "vehicle":
            vid = self.selected[1]
            wps = self._vehicle_routes.get(str(vid))
            if wps:
                prog = int(self._route_progress.get(str(vid), 0))
                for i_wp, (lane_id, s) in enumerate(wps):
                    lane = self.world.lanes.get(lane_id)
                    if lane is None:
                        continue
                    x, y = point_at_s(lane.polyline, float(s))
                    sx, sy = self.vp.world_to_screen(x, y, cw, ch)
                    reached = (i_wp < prog)
                    fill = "#ff3b3b" if reached else "#ffd166"
                    r = 6
                    self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=fill, outline="#111", width=1, tags=("route_pt",))
                    # show waypoint order number; keep it visible even after reaching
                    txt_col = "#ffffff" if reached else "#111111"
                    self.canvas.create_text(sx, sy, text=str(i_wp + 1), fill=txt_col, font=("Helvetica", 9, "bold"), tags=("route_pt",))
        # Stuck-vehicle markers — drawn from persistent positions so they remain
        # visible for the entire simulation run rather than vanishing after 200
        # new events have arrived.
        for (wx, wy) in self._persistent_stuck_positions:
            sx, sy = self.vp.world_to_screen(wx, wy, cw, ch)
            self.canvas.create_line(sx - 10, sy - 10, sx + 10, sy + 10, fill="#ff3b3b", width=3)
            self.canvas.create_line(sx - 10, sy + 10, sx + 10, sy - 10, fill="#ff3b3b", width=3)

        # ── Scene name overlay (top-centre of canvas) ────────────────────────
        _sn = getattr(self, "_scene_name", "Untitled")
        _tx = cw // 2
        _ty = 18
        # Semi-transparent pill background: draw a rounded rectangle via two
        # overlapping rectangles (Tk Canvas doesn't natively support rounded
        # corners) — use a solid dark fill which blends against the dark background.
        _pad_x, _pad_y = 14, 5
        _font_sn = ("Helvetica", 13, "bold")
        # Measure approximate text width (heuristic: ~8 px per char at size 13)
        _tw = len(_sn) * 8 + _pad_x * 2
        self.canvas.create_rectangle(
            _tx - _tw // 2, _ty - _pad_y - 10,
            _tx + _tw // 2, _ty + _pad_y + 4,
            fill="#1a1f2b", outline="#2e3a50", width=1,
        )
        self.canvas.create_text(
            _tx, _ty - 3,
            text=_sn,
            fill="#e0e8f4",
            font=_font_sn,
            anchor="center",
        )

    # ---------------- Data handling ----------------
    def _ui_pump(self):
        pulled = 0
        # Limit per-tick work so the UI stays responsive even when the sim is
        # producing lots of events.
        MAX_PULL = 800
        while pulled < MAX_PULL:
            try:
                ev = self.ev_q.get_nowait()
            except Empty:
                break
            pulled += 1
            self._all_events.append(ev)
            # Keep persistent marker state in sync as events arrive so that
            # the render loop can use these sets directly instead of re-scanning
            # the rolling _all_events window every frame.
            _ev_topic = getattr(ev, 'topic', '')
            _ev_p = getattr(ev, 'payload', None) or {}
            if _ev_topic == 'world.collision':
                for _vid_c in _ev_p.get('vehicle_ids', []) or []:
                    self._persistent_collided_ids.add(str(_vid_c))
            elif _ev_topic == 'world.route_complete':
                _vid_c = _ev_p.get('vehicle_id')
                if _vid_c:
                    self._persistent_route_complete_ids.add(str(_vid_c))
            elif _ev_topic == 'world.vehicle_stuck':
                _x_s, _y_s = _ev_p.get('x'), _ev_p.get('y')
                if _x_s is not None and _y_s is not None:
                    _pos = (float(_x_s), float(_y_s))
                    if _pos not in self._persistent_stuck_positions:
                        self._persistent_stuck_positions.append(_pos)
            if _ev_topic == 'world.route_waypoint_reached':
                p = getattr(ev, 'payload', None) or {}
                vid = p.get('vehicle_id')
                if vid:
                    # Advance progress counter (robust to slight numeric drift)
                    cur = int(self._route_progress.get(str(vid), 0))
                    self._route_progress[str(vid)] = cur + 1

        if pulled > 0 and self.nb.index("current") == 0:
            self._request_render()

        # Update timeline slider
        if self._all_events:
            # find latest timestamp
            latest_t = None
            for ev in reversed(self._all_events[-200:]):
                p = getattr(ev, 'payload', None) or {}
                if 't' in p:
                    latest_t = float(p.get('t', 0.0))
                    break
            if latest_t is not None and latest_t > self._time_max:
                self._time_max = latest_t
                try:
                    self.time_scale.configure(to=self._time_max)
                except Exception:
                    pass
            # If running, follow live time; if stopped and not scrubbing, keep label in sync
            if self._running and not self._scrubbing:
                self.view_time = None
                self.time_var.set(self._time_max)
            self._time_label.configure(text=f"{float(self.time_var.get()):.2f}s")

        # If backlog exists, pump sooner to drain without freezing.
        try:
            has_more = not self.ev_q.empty()
        except Exception:
            has_more = False
        self.after(5 if has_more else 30, self._ui_pump)

    def _clear_table(self, tv):
        """Delete all data rows, preserving any 'col_desc' description row."""
        for iid in tv.get_children():
            if "col_desc" not in tv.item(iid, "tags"):
                tv.delete(iid)

    def _refresh_das_table(self):
        """Re-render DAS table according to the selected sensor filter."""
        try:
            self._clear_table(self.tv_das)
        except Exception:
            return

        rows = getattr(self, "_das_rows_full", [])
        sel = "All"
        try:
            if hasattr(self, "_das_sensor_var"):
                sel = self._das_sensor_var.get()
        except Exception:
            sel = "All"

        for row in rows:
            if sel == "All" or row[1] == sel:
                self.tv_das.insert("", "end", values=row)

    def _finalize_dataset(self):
        if not self._all_events:
            self.data_status.set("No events captured.")
            return

        self._data_ready = True
        self.data_status.set(f"Dataset ready: {len(self._all_events)} events. Use subtabs to inspect and export.")
        self._populate_data_tables()

    def _populate_data_tables(self):
        self._clear_table(self.tv_summary)
        self._clear_table(self.tv_veh)
        self._clear_table(self.tv_gps)
        self._clear_table(self.tv_cam)
        self._clear_table(self.tv_das)
        self._clear_table(self.tv_kalman)
        self._clear_table(self.tv_rmse)
        self._clear_table(self.tv_anom)
        # Phase 2b: Tracks tab is rebuilt from the track.* ring buffer.
        # This mirrors the clear-and-rebuild pattern used for every other
        # data table so the two surfaces stay in lock-step.
        try:
            self._clear_table(self.tv_tracks)
        except Exception:
            pass

        # Keep full DAS rows so we can filter by sensor_id in the UI.
        self._das_rows_full = []

        # Build GID lookup for sensor-tab annotation.  Safe fallback to empty
        # dicts so a TrackManager failure never breaks data-table population.
        try:
            _sensor_lookup, _vid_to_gid = build_sensor_gid_lookup(self._all_events)
        except Exception:
            _sensor_lookup, _vid_to_gid = {}, {}

        cnt = {}
        for ev in self._all_events:
            cnt[ev.topic] = cnt.get(ev.topic, 0) + 1

        self.tv_summary.insert("", "end", values=("events_total", str(len(self._all_events))))
        for k in sorted(cnt.keys()):
            self.tv_summary.insert("", "end", values=(k, str(cnt[k])))

        for ev in self._all_events:
            p = ev.payload
            if ev.topic == "world.vehicle_state":
                _vid = p.get("vehicle_id", "")
                self.tv_veh.insert("", "end", values=(
                    f"{p.get('t', 0.0):.2f}",
                    _vid,
                    _vid_to_gid.get(str(_vid), ""),
                    p.get("lane_id", ""),
                    f"{p.get('x', 0.0):.2f}",
                    f"{p.get('y', 0.0):.2f}",
                    f"{p.get('v', 0.0):.2f}",
                    f"{p.get('heading_rad', 0.0):.3f}",
                    f"{p.get('a_long_mps2', 0.0):.3f}",
                    f"{p.get('ax_world_mps2', 0.0):.3f}",
                    f"{p.get('ay_world_mps2', 0.0):.3f}",
                ))
            elif ev.topic == "sensor.gps":
                pass
            elif ev.topic == "sensor.camera":
                pass
            elif ev.topic == "sensor.das":
                # DAS rows are derived later so we can compute along-fiber acceleration
                # from the DAS speed estimates in a consistent way.
                pass
            elif ev.topic == "world.vehicle_stuck":
                det = f"vehicle={p.get('vehicle_id')} node={p.get('node_id')} lane={p.get('lane_id')}"
                self.tv_anom.insert("", "end", values=(f"{p.get('t', 0.0):.2f}", "vehicle_stuck", det))
            elif ev.topic == "world.collision":
                # Anomaly: a collision pair was detected.  Render in the
                # anomalies tab so the user can audit which (follower, leader)
                # ids and at what time the impact happened.
                _vids = p.get("vehicle_ids") or [p.get("follower_id"), p.get("leader_id")]
                det = (
                    f"pair={'+'.join(str(x) for x in _vids if x)} "
                    f"lane={p.get('lane_id', '')} "
                    f"rel_v={float(p.get('rel_speed_mps', 0.0)):.2f}"
                )
                self.tv_anom.insert("", "end", values=(f"{p.get('t', 0.0):.2f}", "collision", det))
            elif ev.topic == "world.route_complete":
                det = (
                    f"vehicle={p.get('vehicle_id')} "
                    f"lane={p.get('lane_id', '')} s={float(p.get('s', 0.0)):.2f}"
                )
                self.tv_anom.insert("", "end", values=(f"{p.get('t', 0.0):.2f}", "route_complete", det))

        # Kalman fusion table (best-effort fusion using the legacy 6-state model
        # from the previous project: [x, vx, ax, y, vy, ay]).
        for row in self._derive_sensor_acc_rows("sensor.gps", _sensor_lookup):
            self.tv_gps.insert("", "end", values=row)
        for row in self._derive_sensor_acc_rows("sensor.camera", _sensor_lookup):
            self.tv_cam.insert("", "end", values=row)
        self._das_rows_full = self._derive_das_rows(_sensor_lookup)
        self._refresh_das_table()
        self._plot_das_trajectories()

        # Phase 5: Kalman rows are produced by the tracker-driven
        # build_kalman_rows_tracked.  Phase 7: the row schema widened by
        # one column — ``global_track_id`` is now emitted right after
        # ``t`` so the tracker identity is visible next to the oracle
        # ground-truth id.  The tv_kalman column tuple above was updated
        # to match; XLSX header propagation via _tv_to_data(tv_kalman)
        # carries the new column automatically.  Stripping the gid
        # column yields the legacy 12-column schema byte-for-byte on
        # Phase-1 (1:1 vid ↔ gid) scenes — asserted by the
        # test_tracked_single_vehicle_converges_to_legacy_exactly and
        # test_tracked_two_vehicles_no_contamination tests.
        for row in build_kalman_rows_tracked(self._all_events):
            self.tv_kalman.insert("", "end", values=row)

        for row in self._build_rmse_rows():
            self.tv_rmse.insert("", "end", values=row)

        # Update DAS sensor filter dropdown (All + sorted sensor ids)
        try:
            if hasattr(self, "_das_sensor_cb"):
                sids = sorted({r[1] for r in self._das_rows_full if r[1]})
                vals = ("All",) + tuple(sids)
                self._das_sensor_cb.configure(values=vals)
                if self._das_sensor_var.get() not in vals:
                    self._das_sensor_var.set("All")
        except Exception:
            pass

        # Phase 2b: populate the Tracks tab from the track.* ring buffer.
        # This is completely separate from self._all_events — the tracker
        # side-channel has its own bounded buffer.  If something in here
        # fails we swallow it: the other tables are unaffected.
        try:
            self._populate_tracks_table()
        except Exception:
            pass

        # Phase 3: populate the Tracking-Diagnostics tab from the
        # TrackManager directly (summary + per-track rows).  Wrapped in
        # its own try/except for the same robustness reason as above.
        try:
            self._populate_diag_tab()
        except Exception:
            pass

    # Minimum number of tracker updates a track must have before it is shown
    # in the Tracks table and included in the Excel export.  Tracks below this
    # threshold are single-sensor ghost detections (typically one DAS fiber hit)
    # and are not meaningful for analysis.  Raise this value if you still see
    # spurious 1-2-point tracks; lower it (min 1) if you want to keep everything.
    MIN_TRACK_UPDATES: int = 5

    def _populate_tracks_table(self):
        """Render the contents of ``self._track_buffer`` into ``self.tv_tracks``.

        Only ``track.update`` payloads produce rows (they carry the full
        kinematic snapshot); ``track.birth`` events are structural markers
        and are intentionally not rendered as data rows.  Events are
        inserted in arrival order, which for a synchronous bus is also
        time order.

        Ghost-track filter: tracks with fewer than ``MIN_TRACK_UPDATES``
        updates are silently suppressed.  These are typically single DAS fiber
        hits that the tracker could not associate with an existing track.
        They add thousands of rows to the export but carry no useful
        trajectory information.
        """
        buf = getattr(self, "_track_buffer", None)
        if buf is None:
            return
        # Snapshot — iterate over a list to avoid surprises if new events
        # arrive during rendering.
        events = list(buf)

        # Pass 1 — count updates per track ID so we can filter below.
        from collections import Counter as _Counter
        _update_counts = _Counter()
        for _ev in events:
            if getattr(_ev, "topic", "") == "track.update":
                _p = getattr(_ev, "payload", None) or {}
                if isinstance(_p, dict):
                    _update_counts[str(_p.get("global_track_id", ""))] += 1

        min_upd = int(getattr(self, "MIN_TRACK_UPDATES", 5))

        # Pass 2 — render only tracks that meet the threshold.
        for ev in events:
            topic = getattr(ev, "topic", "")
            if topic != "track.update":
                continue
            p = getattr(ev, "payload", None) or {}
            if not isinstance(p, dict):
                continue
            # Ghost-track filter
            if _update_counts.get(str(p.get("global_track_id", "")), 0) < min_upd:
                continue
            self.tv_tracks.insert("", "end", values=(
                f"{float(p.get('t', 0.0) or 0.0):.2f}",
                str(p.get("global_track_id", "")),
                str(p.get("vehicle_id_oracle", "")),
                str(p.get("source", "")),
                f"{float(p.get('x', 0.0) or 0.0):.2f}",
                f"{float(p.get('y', 0.0) or 0.0):.2f}",
                f"{float(p.get('v', 0.0) or 0.0):.2f}",
                str(p.get("segment_id", "")),
                str(p.get("lane_id", "")),
                str(bool(p.get("tentative", False))),
                int(p.get("n_gps", 0) or 0),
                int(p.get("n_cam", 0) or 0),
                int(p.get("n_das", 0) or 0),
                int(p.get("n_state", 0) or 0),
            ))

    def _populate_diag_tab(self):
        """Render the Phase 3 Tracking-Diagnostics tab.

        Reads everything it needs from :class:`TrackManager` directly
        (``summary()`` and ``diagnostic_rows()``) — not from the ring
        buffer — because the manager is the authoritative aggregated
        view.  Empty trackers produce empty tables (no placeholder
        rows, per the Phase 3 design agreement).

        This is wrapped in a broad try/except by ``_populate_data_tables``
        so a tracker failure here never takes down the other tables.
        """
        tm = getattr(self, "_track_manager", None)
        tv_sum = getattr(self, "tv_diag_summary", None)
        tv_rows = getattr(self, "tv_diag", None)
        # Any of these may be absent if import failed or the tab was
        # not built — bail silently.
        if tv_sum is not None:
            try:
                self._clear_table(tv_sum)
            except Exception:
                pass
        if tv_rows is not None:
            try:
                self._clear_table(tv_rows)
            except Exception:
                pass
        if tm is None:
            return

        # --- top summary pane -------------------------------------------
        if tv_sum is not None:
            try:
                summary = tm.summary() or {}
            except Exception:
                summary = {}
            # Stable, curated display order (keeps the pane readable).
            key_order = (
                "n_tracks", "n_confirmed", "n_tentative",
                "error_count", "next_track_ix", "graph_size",
                "n_published_birth", "n_published_update",
                "bus_attached",
            )
            seen = set()
            for k in key_order:
                if k in summary:
                    tv_sum.insert("", "end", values=(k, str(summary[k])))
                    seen.add(k)
            # Any extra keys introduced later show up at the bottom.
            for k, v in summary.items():
                if k not in seen:
                    tv_sum.insert("", "end", values=(k, str(v)))

        # --- per-track rows ---------------------------------------------
        if tv_rows is None:
            return
        try:
            rows = tm.diagnostic_rows() or []
        except Exception:
            rows = []
        # Use the authoritative column order from the tracker module so
        # we cannot drift out of sync with ``diagnostic_rows()``.
        try:
            from ..tracking import TrackManager as _TM
            cols = tuple(_TM.DIAGNOSTIC_COLUMNS)
        except Exception:
            cols = tuple(tv_rows["columns"])

        # Normalise t_born / t_last to simulation-relative seconds so the
        # displayed values start near 0 instead of showing a Unix epoch
        # (e.g. 1 777 744 061 s).  t0 is the earliest birth time seen; all
        # timestamps are shown as offsets from that point.
        t0 = 0.0
        if rows:
            try:
                t0 = min(float(r.get("t_born", 0) or 0) for r in rows)
            except Exception:
                t0 = 0.0

        def _fmt(col, val):
            # Cheap, GUI-safe formatting.  Never raise: fall back to str().
            try:
                if col in ("t_born", "t_last"):
                    return f"{float(val) - t0:.2f}"
                if col == "duration_s":
                    return f"{float(val):.2f}"
                if col in (
                    "n_gps", "n_cam", "n_das", "n_state",
                    "n_segments", "hypothesis_count",
                    "id_switches", "n_updates_total",
                ):
                    return str(int(val))
                return "" if val is None else str(val)
            except Exception:
                return "" if val is None else str(val)

        for row in rows:
            values = tuple(_fmt(c, row.get(c, "")) for c in cols)
            tv_rows.insert("", "end", values=values)

    # ──────────────────────────────────────────────────────────────────────────
    # ANOMALY INTELLIGENCE TAB
    # ──────────────────────────────────────────────────────────────────────────

    def _build_anomaly_intel_tab(self, parent):
        """Build the Anomaly Intelligence tab with matplotlib charts and score cards.

        Layout
        ------
        [control bar: Analyze button + status]
        ┌─────────────────┬──────────────────────────────────────┐
        │ VEHICLE SCORES  │  Notebook: Heatmap / Scores / Dist   │
        │ (Treeview)      │  (matplotlib FigureCanvasTkAgg)      │
        │ + model info    │                                      │
        └─────────────────┴──────────────────────────────────────┘
        """
        import os
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # ── Control bar ──────────────────────────────────────────────────────
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        self._ai_btn = ttk.Button(
            bar, text="🔬  Analyze Simulation", command=self._prompt_and_run_analysis
        )
        self._ai_btn.pack(side="left", padx=(0, 6))

        self._ai_load_csv_btn = ttk.Button(
            bar, text="📂  Load Feature CSV…", command=self._load_csv_and_score
        )
        self._ai_load_csv_btn.pack(side="left", padx=(0, 12))

        self._ai_export_btn = ttk.Button(
            bar, text="📄  Export Report…", command=self._ai_export_report,
            state="disabled"
        )
        self._ai_export_btn.pack(side="left", padx=(0, 12))

        self._ai_status = tk.StringVar(value="Ready — click Analyze after running a simulation.")
        ttk.Label(bar, textvariable=self._ai_status, foreground="#555555").pack(
            side="left", fill="x", expand=True
        )

        # Model path probe — prefer model_live.pkl (version-agnostic numpy format),
        # fall back to model.pkl if live model not yet trained.
        def _find_model():
            for depth in (3, 4):
                base = Path(__file__).resolve().parents[depth] / "anomaly_model" / "outputs"
                live = base / "model_live.pkl"
                if live.exists():
                    return live
                classic = base / "model.pkl"
                if classic.exists():
                    return classic
            return None
        self._ai_model_path = _find_model()

        # ── Main split pane ──────────────────────────────────────────────────
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

        # ── Left: vehicle score table + model info ───────────────────────────
        left = ttk.Frame(paned, width=260)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=2)   # score table
        left.rowconfigure(4, weight=1)   # label separator
        left.rowconfigure(5, weight=3)   # feature details table
        paned.add(left, weight=1)

        ttk.Label(left, text="Vehicle Anomaly Scores",
                  font=("Helvetica", 11, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 2))

        score_frame = ttk.Frame(left)
        score_frame.grid(row=1, column=0, sticky="nsew", padx=4)
        score_frame.columnconfigure(0, weight=1)
        score_frame.rowconfigure(0, weight=1)

        ai_cols = ("vehicle", "z_score", "status", "top_feature")
        self.tv_ai_scores = ttk.Treeview(
            score_frame, columns=ai_cols, show="headings", height=10
        )
        for col, txt, w, anchor in [
            ("vehicle",     "vehicle",     70,  "w"),
            ("z_score",     "z_score",     70,  "e"),
            ("status",      "status",      80,  "center"),
            ("top_feature", "top_feature", 200, "w"),
        ]:
            self.tv_ai_scores.heading(col, text=txt)
            self.tv_ai_scores.column(col, width=w, minwidth=30, anchor=anchor, stretch=False)
        self.tv_ai_scores.grid(row=0, column=0, sticky="nsew")
        vs_ai = ttk.Scrollbar(score_frame, orient="vertical", command=self.tv_ai_scores.yview)
        vs_ai.grid(row=0, column=1, sticky="ns")
        self.tv_ai_scores.configure(yscrollcommand=vs_ai.set)

        # Tag colours
        self.tv_ai_scores.tag_configure("anomaly",  foreground="#cc2200", font=("TkDefaultFont", 9, "bold"))
        self.tv_ai_scores.tag_configure("marginal", foreground="#cc7700")
        self.tv_ai_scores.tag_configure("normal",   foreground="#226600")
        self.tv_ai_scores.tag_configure("header",   foreground="#888888", font=("TkDefaultFont", 9, "italic"))

        # Model info panel
        self._ai_info_var = tk.StringVar(value="Model: not loaded")
        info_lf = ttk.LabelFrame(left, text="Model Configuration")
        info_lf.grid(row=2, column=0, sticky="ew", padx=4, pady=(6, 2))
        ttk.Label(info_lf, textvariable=self._ai_info_var,
                  wraplength=230, justify="left", foreground="#333333").pack(
            anchor="w", padx=6, pady=4
        )

        # Verdict banner
        self._ai_verdict_var = tk.StringVar(value="")
        self._ai_verdict_lbl = ttk.Label(
            left, textvariable=self._ai_verdict_var,
            font=("Helvetica", 13, "bold"), anchor="center"
        )
        self._ai_verdict_lbl.grid(row=3, column=0, sticky="ew", padx=6, pady=(6, 4))

        # ── Feature Details catalog table (left panel, rows 4-5) ────────────
        # Shows: feature name | what it measures | typical range
        ttk.Label(left, text="Feature Details",
                  font=("Helvetica", 10, "bold")).grid(
            row=4, column=0, sticky="w", padx=6, pady=(6, 1))

        insp_frame = ttk.Frame(left)
        insp_frame.grid(row=5, column=0, sticky="nsew", padx=4, pady=(0, 4))
        insp_frame.columnconfigure(0, weight=1)
        insp_frame.rowconfigure(0, weight=1)

        _cat_cols = ("feature", "anomaly_type", "description", "output_range")
        self._ai_catalog_table = ttk.Treeview(
            insp_frame, columns=_cat_cols, show="headings", height=6
        )
        for _col, _txt, _w, _anc in [
            ("feature",      "Feature",          140, "w"),
            ("anomaly_type", "Anomaly",          130, "w"),
            ("description",  "What it measures", 200, "w"),
            ("output_range", "Typical range",    100, "center"),
        ]:
            self._ai_catalog_table.heading(_col, text=_txt)
            self._ai_catalog_table.column(_col, width=_w, minwidth=30,
                                          anchor=_anc, stretch=False)

        self._ai_catalog_table.tag_configure("row_odd",  background="#f7f7f7")
        self._ai_catalog_table.tag_configure("row_even", background="#ffffff")

        _vs_cat = ttk.Scrollbar(insp_frame, orient="vertical",
                                command=self._ai_catalog_table.yview)
        _hs_cat = ttk.Scrollbar(insp_frame, orient="horizontal",
                                command=self._ai_catalog_table.xview)
        self._ai_catalog_table.configure(
            yscrollcommand=_vs_cat.set, xscrollcommand=_hs_cat.set)
        self._ai_catalog_table.grid(row=0, column=0, sticky="nsew")
        _vs_cat.grid(row=0, column=1, sticky="ns")
        _hs_cat.grid(row=1, column=0, sticky="ew")

        # ── Right: matplotlib charts in a sub-notebook ───────────────────────
        right = ttk.Frame(paned)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        paned.add(right, weight=3)

        if _MPL_OK:
            chart_nb = ttk.Notebook(right)
            chart_nb.grid(row=0, column=0, sticky="nsew")

            # Tab A — Z-Score heatmap
            tab_heat = ttk.Frame(chart_nb)
            chart_nb.add(tab_heat, text="  Z-Score Heatmap  ")
            tab_heat.columnconfigure(0, weight=1)
            tab_heat.rowconfigure(0, weight=1)
            self._ai_fig_heat = Figure(figsize=(7, 4), dpi=90)
            self._ai_fig_heat.patch.set_facecolor("#f5f5f5")
            self._ai_canvas_heat = FigureCanvasTkAgg(self._ai_fig_heat, master=tab_heat)
            self._ai_canvas_heat.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            # Tab B — Anomaly score bars (Z + IF)
            tab_bars = ttk.Frame(chart_nb)
            chart_nb.add(tab_bars, text="  Score Ranking  ")
            tab_bars.columnconfigure(0, weight=1)
            tab_bars.rowconfigure(0, weight=1)
            self._ai_fig_bars = Figure(figsize=(7, 4), dpi=90)
            self._ai_fig_bars.patch.set_facecolor("#f5f5f5")
            self._ai_canvas_bars = FigureCanvasTkAgg(self._ai_fig_bars, master=tab_bars)
            self._ai_canvas_bars.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            # Tab C — Feature distribution (normal vs this scenario)
            tab_dist = ttk.Frame(chart_nb)
            chart_nb.add(tab_dist, text="  Feature Distributions  ")
            tab_dist.columnconfigure(0, weight=1)
            tab_dist.rowconfigure(0, weight=3)   # plots get most of the height
            tab_dist.rowconfigure(1, weight=1)   # table below

            self._ai_fig_dist = Figure(figsize=(7, 4), dpi=90)
            self._ai_fig_dist.patch.set_facecolor("#f5f5f5")
            self._ai_canvas_dist = FigureCanvasTkAgg(self._ai_fig_dist, master=tab_dist)
            self._ai_canvas_dist.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            # ── Measured-values table (below plots) ──────────────────────────
            # Shows: Feature | Vehicle | Measured value | Z-score | Normal range | Status
            dist_tbl_frame = ttk.Frame(tab_dist)
            dist_tbl_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))
            dist_tbl_frame.columnconfigure(0, weight=1)
            dist_tbl_frame.rowconfigure(0, weight=1)

            _dist_cols = ("feature", "vehicle", "value", "z_score", "normal_range", "status", "anomaly_type")
            self._ai_dist_table = ttk.Treeview(
                dist_tbl_frame, columns=_dist_cols, show="headings", height=5
            )
            for _col, _txt, _w, _anc in [
                ("feature",      "Feature",           170, "w"),
                ("vehicle",      "Vehicle",            70, "center"),
                ("value",        "Measured value",    120, "e"),
                ("z_score",      "Z-score (σ)",        85, "e"),
                ("normal_range", "Normal range (±2σ)", 150, "center"),
                ("status",       "Status",             80, "center"),
                ("anomaly_type", "Anomaly",            140, "w"),
            ]:
                self._ai_dist_table.heading(_col, text=_txt)
                self._ai_dist_table.column(_col, width=_w, minwidth=30,
                                           anchor=_anc, stretch=False)

            self._ai_dist_table.tag_configure("anomaly",  foreground="#cc2200",
                                              font=("TkDefaultFont", 8, "bold"))
            self._ai_dist_table.tag_configure("marginal", foreground="#cc7700")
            self._ai_dist_table.tag_configure("normal",   foreground="#226600")

            _vs_dist = ttk.Scrollbar(dist_tbl_frame, orient="vertical",
                                     command=self._ai_dist_table.yview)
            _hs_dist = ttk.Scrollbar(dist_tbl_frame, orient="horizontal",
                                     command=self._ai_dist_table.xview)
            self._ai_dist_table.configure(
                yscrollcommand=_vs_dist.set, xscrollcommand=_hs_dist.set)
            self._ai_dist_table.grid(row=0, column=0, sticky="nsew")
            _vs_dist.grid(row=0, column=1, sticky="ns")
            _hs_dist.grid(row=1, column=0, sticky="ew")

            # Tab D — CUSUM temporal anomaly curve
            tab_cusum = ttk.Frame(chart_nb)
            chart_nb.add(tab_cusum, text="  CUSUM  ")
            tab_cusum.columnconfigure(0, weight=1)
            tab_cusum.rowconfigure(0, weight=1)
            self._ai_fig_cusum = Figure(figsize=(7, 4), dpi=90)
            self._ai_fig_cusum.patch.set_facecolor("#f5f5f5")
            self._ai_canvas_cusum = FigureCanvasTkAgg(self._ai_fig_cusum, master=tab_cusum)
            self._ai_canvas_cusum.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        else:
            self._ai_fig_heat = self._ai_fig_bars = self._ai_fig_dist = self._ai_fig_cusum = None
            self._ai_canvas_heat = self._ai_canvas_bars = self._ai_canvas_dist = self._ai_canvas_cusum = None
            ttk.Label(right, text="matplotlib not available — install it for charts.",
                      foreground="#aa0000").pack(padx=20, pady=20)

        # Load and cache model info
        self._ai_model_data = {}
        self._ai_df_normal = None
        self._ai_analysis_result = None
        self._ai_load_model()

    def _ai_load_model(self):
        """Load anomaly model into instance cache (best-effort).

        Supports two formats:

        Format v2  (model_live.pkl) — version-agnostic, numpy arrays only:
            {
                "format_version": 2,
                "model_type":     "zscore_max",
                "feature_cols":   [str, ...],
                "scaler_mean":    np.ndarray,
                "scaler_std":     np.ndarray,
                "feature_medians":np.ndarray,
                "zscore_threshold": float,
                "contamination":  float,
            }

        Format v1  (model.pkl) — sklearn Pipeline (older, may fail on version mismatch).

        After loading, self._ai_model_data exposes a normalised dict used by
        _ai_extract_and_score:
            "feature_cols"      list[str]
            "scaler_mean"       np.ndarray  (v2) or None  (v1 uses scaler object)
            "scaler_std"        np.ndarray  (v2) or None
            "feature_medians"   np.ndarray  or None
            "scaler"            object      (v1 sklearn scaler) or None
            "model"             object      (v1 sklearn model) or None
            "threshold"         float
            "contamination"     float
            "format_version"    int
        """
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            self._ai_info_var.set("numpy / pandas not installed.")
            return

        if self._ai_model_path and self._ai_model_path.exists():
            try:
                import pickle
                with open(self._ai_model_path, "rb") as fh:
                    raw = pickle.load(fh)

                fmt = int(raw.get("format_version", 1))

                if fmt >= 2:
                    # ── Format v2: pure numpy, no sklearn dependency ──────────
                    bundle = {
                        "format_version":   fmt,
                        "feature_cols":     raw["feature_cols"],
                        "scaler_mean":      np.asarray(raw["scaler_mean"],      dtype=float),
                        "scaler_std":       np.asarray(raw["scaler_std"],       dtype=float),
                        "feature_medians":  np.asarray(raw.get("feature_medians",
                                            raw["scaler_mean"]), dtype=float),
                        "scaler":           None,
                        "model":            None,
                        "threshold":        float(raw.get("zscore_threshold", 7.76)),
                        "contamination":    float(raw.get("contamination", 0.01)),
                        "model_type":       raw.get("model_type", "zscore_max"),
                        "n_train":          int(raw.get("n_train", 0)),
                    }
                    model_label = f"Z-Score (live, {bundle['n_train']} training vehicles)"
                else:
                    # ── Format v1: sklearn Pipeline ───────────────────────────
                    bundle = dict(raw)
                    if "pipeline" in raw and "scaler" not in raw:
                        try:
                            pipeline = raw["pipeline"]
                            steps    = pipeline.named_steps
                            bundle["scaler"] = steps.get("scaler")
                            bundle["model"]  = steps.get("iforest") or steps.get("rf")
                        except Exception:
                            bundle["scaler"] = None
                            bundle["model"]  = None
                    if "zscore_threshold" in raw and "threshold" not in raw:
                        bundle["threshold"] = raw["zscore_threshold"]
                    bundle.setdefault("threshold",    5.47)
                    bundle.setdefault("contamination", 0.15)
                    bundle.setdefault("scaler_mean",  None)
                    bundle.setdefault("scaler_std",   None)
                    bundle.setdefault("feature_medians", None)
                    bundle["format_version"] = 1
                    model_label = "Isolation Forest (legacy)"

                self._ai_model_data = bundle
                feat_cols = bundle.get("feature_cols", [])
                thr  = bundle["threshold"]
                cont = bundle["contamination"]
                self._ai_info_var.set(
                    f"Model: {model_label}\n"
                    f"Features: {len(feat_cols)}\n"
                    f"Z-threshold: {thr:.3f} σ\n"
                    f"Contamination: {cont:.1%}\n"
                    f"Path: …/{self._ai_model_path.parent.name}/{self._ai_model_path.name}"
                )
            except Exception as e:
                self._ai_info_var.set(f"Model load error:\n{e}")
        else:
            self._ai_info_var.set(
                "model_live.pkl not found.\nTrain the live model:\n"
                "python anomaly_model/scripts/scenario_scorer.py --train-live"
            )

        # Load normal features for distribution plots
        norm_path = (self._ai_model_path.parent / "features_normal.csv"
                     if self._ai_model_path else None)
        if norm_path and norm_path.exists():
            try:
                self._ai_df_normal = pd.read_csv(norm_path)
            except Exception:
                self._ai_df_normal = None

    def _load_csv_and_score(self):
        """Load a previously-exported feature CSV and run anomaly analysis on it.

        Expected format: one row per vehicle track, one column per feature.
        The file must contain a 'track_id' (or 'global_track_id') column plus
        numerical feature columns matching the model's expected feature set.
        Produced by:  python anomaly_model/scripts/scenario_scorer.py --extract ...
                 or:  the GUI's own 'Save Reports' → features_anomaly.csv export.
        """
        import tkinter.filedialog as _fd
        try:
            import pandas as pd
        except ImportError:
            self._ai_status.set("pandas not installed — cannot load CSV.")
            return

        csv_path = _fd.askopenfilename(
            title="Select a feature CSV to analyse",
            filetypes=[
                ("Feature CSV", "*.csv"),
                ("All files",   "*.*"),
            ],
            initialdir=str(Path(__file__).resolve().parents[3] /
                           "anomaly_model" / "outputs"),
        )
        if not csv_path:
            return

        self._ai_status.set(f"Loading {Path(csv_path).name} …")
        self._ai_btn.configure(state="disabled")
        self._ai_load_csv_btn.configure(state="disabled")

        def _do():
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    self.after(0, lambda: self._ai_status.set(
                        "CSV is empty — nothing to analyse."))
                    return

                # Normalise track_id column
                if "track_id" not in df.columns:
                    if "global_track_id" in df.columns:
                        df["track_id"] = df["global_track_id"]
                    else:
                        df["track_id"] = [f"V{i:03d}" for i in range(len(df))]

                # No live timeseries from a CSV — temporal chart will show
                # "no time-series data" placeholders for each vehicle.
                timeseries_by_vid: dict = {}

                result = self._score_dataframe(df, timeseries_by_vid)
                result["_source"] = Path(csv_path).name  # tag for status bar

                def _show():
                    n_flagged = sum(1 for r in result["rows"] if r.get("flagged"))
                    n_total   = len(result["rows"])
                    src       = result.get("_source", "CSV")
                    self._ai_status.set(
                        f"[{src}]  {n_flagged}/{n_total} flagged  "
                        f"({result.get('n_features_live', '?')}/"
                        f"{result.get('n_features_model', '?')} features)"
                    )
                    self._populate_anomaly_intel(result)
                    self._ai_export_btn.configure(state="normal")

                self.after(0, _show)
            except Exception as exc:
                self.after(0, lambda e=exc: self._ai_status.set(f"Error: {e}"))
            finally:
                self.after(0, lambda: self._ai_btn.configure(state="normal"))
                self.after(0, lambda: self._ai_load_csv_btn.configure(state="normal"))

        import threading
        threading.Thread(target=_do, daemon=True).start()

    def _prompt_and_run_analysis(self):
        """Show a 'save first' dialog, then run anomaly analysis from saved files."""
        events = list(getattr(self, "_all_events", []))
        if not events:
            self._ai_status.set("No simulation data — run a simulation first.")
            return

        last_audit = getattr(self, "_last_audit_dir", None)
        last_xlsx  = getattr(self, "_last_xlsx_export_path", None)
        has_last   = last_audit is not None and Path(last_audit).exists()

        # ── Modal dialog ────────────────────────────────────────────────────
        dlg = tk.Toplevel(self)
        dlg.title("Analyze Simulation — Choose Data Source")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        pad = {"padx": 14, "pady": 6}

        ttk.Label(dlg, text=(
            "For the most accurate anomaly scores, save the full report\n"
            "first so the feature extractor works exactly like the CLI pipeline.\n\n"
            "Choose how to proceed:"
        ), justify="left").pack(anchor="w", **pad)

        ttk.Separator(dlg, orient="horizontal").pack(fill="x", padx=14, pady=2)

        choice = tk.StringVar(value="")

        def _pick(val):
            choice.set(val)
            dlg.destroy()

        btn_opts = {"fill": "x", "padx": 14, "pady": 3}

        if has_last:
            audit_name = Path(last_audit).name
            xlsx_name  = Path(last_xlsx).name if last_xlsx else ""
            auto_label = (f"⚡  Auto — use last saved export\n"
                          f"     {xlsx_name or audit_name}")
            ttk.Button(dlg, text=auto_label,
                       command=lambda: _pick("auto")).pack(**btn_opts)

        ttk.Button(dlg, text="💾  Save Reports Now, then Analyse",
                   command=lambda: _pick("save")).pack(**btn_opts)

        ttk.Button(dlg, text="▶  Analyse from live events (quick, approximate)",
                   command=lambda: _pick("events")).pack(**btn_opts)

        ttk.Separator(dlg, orient="horizontal").pack(fill="x", padx=14, pady=4)
        ttk.Button(dlg, text="✕  Cancel",
                   command=lambda: _pick("cancel")).pack(**btn_opts)

        ttk.Label(dlg, text=(
            "Tip: 'Save Reports' creates the audit folder needed for full\n"
            "feature extraction (Kalman quality, sensor disagreement, etc.)."
        ), foreground="#666666", font=("TkDefaultFont", 8)).pack(anchor="w", padx=14, pady=(0, 8))

        self.wait_window(dlg)

        action = choice.get()
        if action == "cancel" or action == "":
            return

        if action == "auto":
            self._run_anomaly_from_audit_dir(Path(last_audit))
        elif action == "save":
            # Hook: after the next successful export, run analysis automatically.
            self._post_export_cb = self._run_anomaly_from_audit_dir
            self._export_all_tables()
        elif action == "events":
            self._run_anomaly_analysis()

    def _run_anomaly_from_audit_dir(self, audit_dir: "Path"):
        """Run anomaly analysis using files in the tracking_audit folder.

        Uses the same ScenarioLoader that the CLI batch pipeline uses, so
        results are numerically identical to running scenario_scorer.py --score.

        Parameters
        ----------
        audit_dir : Path
            The *_tracking_audit folder written by export_all (contains
            tracking_audit.xlsx, kalman_measurement_audit.xlsx, etc.).
        """
        import sys as _sys
        audit_dir = Path(audit_dir)
        if not audit_dir.exists():
            self._ai_status.set(f"Audit folder not found: {audit_dir}")
            return

        # Make sure anomaly_model is importable
        _repo_root = str(Path(__file__).resolve().parents[3])
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)

        self._ai_btn.configure(state="disabled")
        self._ai_status.set(f"Extracting features from saved files in {audit_dir.name}…")

        def _worker():
            try:
                import numpy as np
                import pandas as pd
                from anomaly_model.feature_extractor import ScenarioLoader

                # Also pass the xlsx export (parent folder may have it)
                xlsx_export = getattr(self, "_last_xlsx_export_path", None)
                scene_json  = getattr(self, "_scene_json_path", None)

                loader = ScenarioLoader(
                    folder=audit_dir,
                    scenario_id=getattr(self, "_scene_name", audit_dir.parent.name),
                    scenario_json=scene_json,
                )
                # If the main xlsx export exists, inject it so xl_* features work
                if xlsx_export and Path(xlsx_export).exists():
                    loader.xl_path = Path(xlsx_export)

                row_dict = loader.extract()
                if isinstance(row_dict, pd.DataFrame):
                    df = row_dict
                else:
                    df = pd.DataFrame([row_dict])

                # Build timeseries from trajectory sheets for CUSUM chart
                timeseries_by_vid: dict = {}
                try:
                    import openpyxl as _oxl
                    from anomaly_model.feature_extractor import _read_track_sheet_from_xlsx
                    wb_path = audit_dir / "tracking_audit.xlsx"
                    if wb_path.exists():
                        _wb = _oxl.load_workbook(wb_path, read_only=True, data_only=True)
                        for sn in _wb.sheetnames:
                            if not sn.startswith("Track "):
                                continue
                            tid = sn[6:].strip()
                            tdf = _read_track_sheet_from_xlsx(wb_path, sn)
                            if tdf is None or "t" not in tdf.columns:
                                continue
                            times  = np.array(pd.to_numeric(tdf["t"],     errors="coerce").values, dtype=float)
                            speeds_col = "v_hat" if "v_hat" in tdf.columns else (
                                         "speed" if "speed" in tdf.columns else None)
                            speeds = (np.array(pd.to_numeric(tdf[speeds_col], errors="coerce").values, dtype=float)
                                      if speeds_col else np.zeros(len(times)))
                            timeseries_by_vid[tid] = (times, speeds, np.array([]), np.array([]))
                        _wb.close()
                except Exception:
                    pass  # timeseries is optional

                result = self._score_dataframe(df, timeseries_by_vid)
                self.after(0, lambda: self._populate_anomaly_intel(result))
            except Exception as exc:
                import traceback as _tb
                msg = f"File-based analysis failed: {exc}"
                print(f"[AnomalyIntel] {msg}\n{_tb.format_exc()}")
                self.after(0, lambda: self._ai_status.set(msg))
            finally:
                self.after(0, lambda: self._ai_btn.configure(state="normal"))
                self.after(0, lambda: self._ai_export_btn.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _score_dataframe(self, df: "pd.DataFrame",
                         timeseries_by_vid: dict) -> dict:
        """Score a pre-extracted feature DataFrame with the loaded model.

        Shared by both _ai_extract_and_score (in-memory path) and
        _run_anomaly_from_audit_dir (file-based path).  Returns the same
        result dict that _populate_anomaly_intel expects.
        """
        import numpy as np
        import pandas as pd

        model_data = self._ai_model_data or {}
        feat_cols  = model_data.get("feature_cols", [])
        _raw_thr   = model_data.get("threshold") or model_data.get("zscore_threshold", 7.76)
        threshold  = float(_raw_thr)

        if not feat_cols:
            skip = {"track_id", "global_track_id", "vehicle_id_oracle",
                    "scenario_id", "label"}
            feat_cols = [c for c in df.columns if c not in skip
                         and pd.api.types.is_numeric_dtype(df[c])]

        n_model  = len(feat_cols)
        fc_avail = [c for c in feat_cols if c in df.columns]

        X_raw = np.full((len(df), n_model), np.nan)
        for ci, col in enumerate(feat_cols):
            if col in df.columns:
                X_raw[:, ci] = pd.to_numeric(df[col], errors="coerce").values

        # NaN imputation
        fmt_ver = int(model_data.get("format_version", 1))
        if fmt_ver >= 2 and model_data.get("feature_medians") is not None:
            fill = np.asarray(model_data["feature_medians"], dtype=float)
        elif self._ai_df_normal is not None:
            fill = np.array([
                float(pd.to_numeric(self._ai_df_normal[c], errors="coerce").median())
                if c in self._ai_df_normal.columns else 0.0
                for c in feat_cols
            ])
        else:
            fill = np.nanmedian(X_raw, axis=0)
            fill = np.where(np.isnan(fill), 0.0, fill)

        X = np.where(np.isnan(X_raw), fill, X_raw)

        # Z-score normalisation
        def _normal_csv_mu_sg():
            if self._ai_df_normal is not None:
                _mu = np.array([
                    float(pd.to_numeric(self._ai_df_normal[c], errors="coerce").mean())
                    if c in self._ai_df_normal.columns else 0.0
                    for c in feat_cols
                ])
                _sg = np.array([
                    float(max(pd.to_numeric(self._ai_df_normal[c], errors="coerce").std(ddof=1), 1e-9))
                    if c in self._ai_df_normal.columns else 1.0
                    for c in feat_cols
                ])
            else:
                _mu = np.nanmean(X_raw, axis=0)
                _sg = np.nanstd(X_raw, axis=0) + 1e-9
            return _mu, _sg

        if model_data.get("scaler_mean") is not None:
            mu = np.asarray(model_data["scaler_mean"], dtype=float)
            sg = np.asarray(model_data["scaler_std"],  dtype=float)
            sg = np.where(sg < 1e-9, 1e-9, sg)
        elif model_data.get("scaler") is not None:
            try:
                Xz_tmp = model_data["scaler"].transform(X)
                mu = np.zeros(n_model); sg = np.ones(n_model)
                X  = Xz_tmp
            except Exception:
                mu, sg = _normal_csv_mu_sg()
        else:
            mu, sg = _normal_csv_mu_sg()

        Xz         = (X - mu) / sg
        z_abs_max  = np.abs(Xz).max(axis=1)
        z_feat_idx = np.argmax(np.abs(Xz), axis=1)

        if_scores = np.zeros(len(X))
        if_model  = model_data.get("model")
        if if_model is not None:
            try:
                if_scores = -if_model.decision_function(X)
            except Exception:
                pass

        rows = df.to_dict("records")
        for row in rows:
            if "track_id" not in row:
                row["track_id"] = row.get("global_track_id", "?")

        for i, row in enumerate(rows):
            row["z_score_max"] = float(z_abs_max[i])
            row["if_score"]    = float(if_scores[i])
            row["flagged"]     = bool(z_abs_max[i] > threshold)
            fidx               = int(z_feat_idx[i])
            row["top_feature"] = feat_cols[fidx] if fidx < len(feat_cols) else "?"

        return {
            "rows":             rows,
            "feat_cols":        feat_cols,
            "z_matrix":         Xz,
            "threshold":        threshold,
            "n_features_live":  len(fc_avail),
            "n_features_model": n_model,
            "timeseries":       timeseries_by_vid,
        }

    def _run_anomaly_analysis(self):
        """Extract live features from current simulation events and score them."""
        events = list(getattr(self, "_all_events", []))
        if not events:
            self._ai_status.set("No simulation data — run a simulation first.")
            return

        self._ai_btn.configure(state="disabled")
        self._ai_status.set("Extracting features from simulation events…")

        def _worker():
            try:
                result = self._ai_extract_and_score(events)
                self.after(0, lambda: self._populate_anomaly_intel(result))
            except Exception as exc:
                import traceback as _tb
                msg = f"Analysis failed: {exc}"
                print(f"[AnomalyIntel] {msg}\n{_tb.format_exc()}")
                self.after(0, lambda: self._ai_status.set(msg))
            finally:
                self.after(0, lambda: self._ai_btn.configure(state="normal"))
                self.after(0, lambda: self._ai_export_btn.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _ai_extract_and_score(self, events):
        """Compute per-vehicle features from in-memory events; score with model.

        Feature extraction is delegated entirely to
        ``anomaly_model.feature_extractor.events_to_feature_dataframe()``,
        which re-uses every ``_features_from_*`` helper from the batch pipeline
        so live and offline features are numerically identical.

        A lightweight separate pass builds ``timeseries_by_vid`` (speed trace +
        inter-vehicle min-distance) for the CUSUM temporal chart only.

        Returns a dict with keys:
          rows      — list of dicts (one per vehicle, with all features + scores)
          feat_cols — list of feature column names used by the model
          z_matrix  — np.ndarray shape (n_vehicles, n_features), signed z-scores
          threshold — float z-score anomaly threshold
          n_features_live  — int, how many model features were computed from live data
          n_features_model — int, total features the model expects
          timeseries       — dict vid → (times, speeds, dist_times, min_dists)
        """
        import numpy as np
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(f"pandas required: {e}")

        # ── 1. Full feature extraction via shared batch-extractor logic ────
        # Add anomaly_model package to sys.path so the import works regardless
        # of how the GUI was launched (installed package, direct run, IDE).
        import sys as _sys
        _repo_root = str(Path(__file__).resolve().parents[3])
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        try:
            from anomaly_model.feature_extractor import events_to_feature_dataframe
        except ImportError as _ie:
            raise RuntimeError(
                f"Cannot import events_to_feature_dataframe from anomaly_model: {_ie}\n"
                f"Expected repo root: {_repo_root}"
            )

        df = events_to_feature_dataframe(
            events,
            scenario_json_path=getattr(self, "_scene_json_path", None),
            scenario_id=getattr(self, "_scene_name", "live_simulation"),
        )

        if df.empty:
            raise RuntimeError("No vehicle_state events found in simulation data.")

        # Expose global_track_id as "track_id" — the GUI uses this key everywhere.
        if "track_id" not in df.columns:
            df.insert(0, "track_id",
                      df["global_track_id"] if "global_track_id" in df.columns
                      else df.index.astype(str))

        all_vids = list(df["track_id"])

        # ── 2. Build CUSUM timeseries (lightweight pass — no feature logic) ─
        # Collect per-vehicle speed traces and inter-vehicle min-distance from
        # world.vehicle_state events.  Nothing here duplicates feature_extractor.
        from collections import defaultdict
        _veh_states: dict = defaultdict(list)
        for ev in events:
            topic = getattr(ev, "topic", "")
            p     = getattr(ev, "payload", None) or {}
            if topic == "world.vehicle_state":
                vid = str(p.get("vehicle_id", ""))
                if vid:
                    _veh_states[vid].append(p)

        # Build sorted (t, x, y, v) lookup per vehicle for distance queries
        vid_xy: dict = {}
        for vid in all_vids:
            states = sorted(_veh_states.get(vid, []),
                            key=lambda p: float(p.get("t", 0) or 0))
            vid_xy[vid] = {
                float(p.get("t", 0) or 0): (
                    float(p.get("x", 0) or 0),
                    float(p.get("y", 0) or 0),
                    float(p.get("v", 0) or 0),
                )
                for p in states
            }

        timeseries_by_vid: dict = {}
        for vid in all_vids:
            states = sorted(_veh_states.get(vid, []),
                            key=lambda p: float(p.get("t", 0) or 0))
            if len(states) < 3:
                continue
            times  = np.array([float(p.get("t", 0) or 0) for p in states])
            speeds = np.array([float(p.get("v", 0) or 0) for p in states])

            # Sample every 5th state for inter-vehicle min-distance (CUSUM proximity)
            sample     = states[::5]
            other_vids = [v for v in all_vids if v != vid]
            dist_times_list: list = []
            min_dists_list:  list = []
            if other_vids and sample:
                for p in sample:
                    t_i = float(p.get("t", 0) or 0)
                    xi  = float(p.get("x", 0) or 0)
                    yi  = float(p.get("y", 0) or 0)
                    step_dists = []
                    for ov in other_vids:
                        oxy = vid_xy.get(ov, {})
                        if not oxy:
                            continue
                        nearest_t = min(oxy.keys(), key=lambda t: abs(t - t_i))
                        xj, yj, _ = oxy[nearest_t]
                        step_dists.append(
                            float(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
                        )
                    if step_dists:
                        dist_times_list.append(t_i)
                        min_dists_list.append(float(np.min(step_dists)))

            timeseries_by_vid[vid] = (
                times,
                speeds,
                np.array(dist_times_list),
                np.array(min_dists_list),
            )

        # ── 3. Score against model (shared helper) ────────────────────────
        return self._score_dataframe(df, timeseries_by_vid)

    def _populate_anomaly_intel(self, result):
        """Update all Anomaly Intel widgets from a scored result dict (main thread)."""
        try:
            import numpy as np
        except ImportError:
            self._ai_status.set("numpy not installed — cannot display results.")
            return

        rows       = result["rows"]
        feat_cols  = result["feat_cols"]
        Xz         = result["z_matrix"]
        threshold  = result["threshold"]
        n_live     = result.get("n_features_live", 0)
        n_model    = result.get("n_features_model", 0)
        timeseries = result.get("timeseries", {})   # vid → (times_arr, speeds_arr)
        self._ai_analysis_result = result

        if not rows:
            self._ai_status.set("No vehicle tracks found in simulation data.")
            return

        n_flagged = sum(1 for r in rows if r.get("flagged"))
        n_total   = len(rows)
        self._ai_status.set(
            f"Analysis complete — {n_flagged}/{n_total} vehicles flagged as anomalous  "
            f"({n_live}/{n_model} model features computed from live data)"
        )

        # ── Verdict ──────────────────────────────────────────────────────────
        if n_flagged > 0:
            top = max(rows, key=lambda r: r.get("z_score_max", 0))
            self._ai_verdict_var.set(
                f"⚠  ANOMALY DETECTED  ({n_flagged}/{n_total})"
            )
            self._ai_verdict_lbl.configure(foreground="#cc2200")
        else:
            self._ai_verdict_var.set("✓  NORMAL SCENARIO")
            self._ai_verdict_lbl.configure(foreground="#226600")

        # ── Score table ───────────────────────────────────────────────────────
        for iid in self.tv_ai_scores.get_children():
            self.tv_ai_scores.delete(iid)

        # Load catalog once for helps_identify weighted voting
        _score_catalog: dict = {}
        try:
            import sys as _sys3
            _rr3 = str(Path(__file__).resolve().parents[3])
            if _rr3 not in _sys3.path:
                _sys3.path.insert(0, _rr3)
            from anomaly_model.make_features_excel import FEATURE_CATALOG_RICH
            _score_catalog = FEATURE_CATALOG_RICH
        except Exception:
            pass

        # Header description row
        self.tv_ai_scores.insert("", "end",
            values=("vehicle", "max |z|σ", "verdict", "peak feature"),
            tags=("header",))

        for ri, row in enumerate(sorted(rows, key=lambda r: r.get("z_score_max", 0), reverse=True)):
            z   = row.get("z_score_max", 0.0)
            flg = row.get("flagged", False)
            tf  = row.get("top_feature", "?")
            if flg:
                verdict, tag = "ANOMALY", "anomaly"
            elif z > threshold * 0.75:
                verdict, tag = "marginal", "marginal"
            else:
                verdict, tag = "normal",  "normal"
            self.tv_ai_scores.insert("", "end",
                values=(row["track_id"], f"{z:.2f}", verdict, tf),
                tags=(tag,))

        if not _MPL_OK:
            return

        # ─── Chart A: Z-Score Heatmap ─────────────────────────────────────────
        fig = self._ai_fig_heat
        fig.clear()
        track_ids = [r["track_id"] for r in rows]

        # Comprehensive category ordering covering all feature families in the
        # 162-feature model.  Unknown features fall through to the end.
        _CATEGORY_ORDER = [
            # ── Collision / Overlap ──────────────────────────────────────────
            "iv_collision_detected", "iv_overlap_frac", "iv_overlap_duration_s",
            "iv_collision_risk_proxy",
            # ── Proximity / TTC ─────────────────────────────────────────────
            "iv_min_dist_m", "iv_mean_min_dist_m", "iv_p10_min_dist_m",
            "iv_close_proximity_frac", "iv_tailgate_proximity_frac",
            "iv_ttc_min_s", "iv_ttc_below_2s_frac",
            "iv_closing_speed_max_mps", "iv_rel_speed_at_min_dist_mps",
            "iv_decel_at_min_dist_mps2",
            # ── Stopping / Stall ────────────────────────────────────────────
            "kin_stopped_frac", "kin_stop_event_count",
            "kin_max_stopped_steps", "kin_max_stopped_duration_s",
            # ── Speed ───────────────────────────────────────────────────────
            "kin_speed_mean_mps", "kin_speed_max_mps", "kin_speed_min_mps",
            "kin_speed_std_mps", "kin_speed_p90_mps", "kin_speed_cv",
            "kin_speed_jump_max_mps",
            "kin_speed_over_limit_frac", "kin_speed_excess_max_mps",
            "kin_speed_excess_mean_mps",
            # ── Acceleration / Jerk ─────────────────────────────────────────
            "kin_accel_mean_abs_mps2", "kin_accel_max_abs_mps2",
            "kin_accel_std_mps2",
            "kin_decel_max_mps2", "kin_decel_event_count", "kin_high_decel_frac",
            "kin_jerk_max_mps3", "kin_jerk_mean_abs_mps3",
            "kin_phys_impossible_v", "kin_phys_impossible_a",
            # ── Lateral / Heading ───────────────────────────────────────────
            "kin_lateral_vel_mean_mps", "kin_lateral_vel_max_mps",
            "kin_lateral_speed_max_mps", "kin_lateral_accel_max_mps2",
            "kin_lateral_dev_mean_m", "kin_lateral_dev_max_m",
            "kin_lateral_dev_std_m", "kin_lateral_peak_to_peak_m",
            "kin_lateral_oscillation_ratio",
            "kin_heading_change_rate_max_rad_per_s", "kin_road_heading_diff_mean_rad",
            # ── Speed vs Traffic ────────────────────────────────────────────
            "iv_speed_excess_over_others_mps", "iv_speed_ratio_to_others",
            "iv_speed_pearson_r_traffic", "iv_speed_proximity_ratio_max",
            "iv_time_headway_mean_s", "iv_n_other_vehicles",
            "iv_others_mean_speed_when_stopped",
            "sc_speed_limit_mps",
            # ── Kalman Filter / Tracking ─────────────────────────────────────
            "kf_pos_err_mean_m", "kf_pos_err_max_m", "kf_pos_err_std_m",
            "kf_pos_err_p90_m", "kf_track_rmse_m", "kf_rolling_rmse_spike_count",
            "kf_sigma_pos_mean_m", "kf_sigma_pos_max_m",
            "kf_consistency_ratio_mean", "kf_consistency_ratio_max",
            "kf_overconfident_frac",
            "kf_sigma_growth_rate_during_gaps_m_per_s", "kf_post_dropout_jump_m",
            # ── Sensor Coverage ──────────────────────────────────────────────
            "cov_total_rows", "cov_pred_only_count", "cov_pred_only_frac",
            "cov_max_consec_pred", "cov_n_dropout_events", "cov_duration_s",
            "cov_das_active_frac", "cov_cam_active_frac", "cov_gps_active_frac",
            "cov_multi_sensor_frac", "cov_single_sensor_frac", "cov_no_sensor_frac",
            "cov_das_cam_overlap_frac", "cov_das_gps_overlap_frac",
            "cov_cam_gps_overlap_frac", "cov_all_sensors_frac",
            "cov_err_das_only_m", "cov_err_cam_only_m",
            # ── Sensor Disagreement ──────────────────────────────────────────
            "dis_das_cam_mean_m", "dis_das_cam_max_m",
            "dis_das_hat_mean_m", "dis_das_hat_max_m",
            "dis_cam_hat_mean_m", "dis_cam_hat_max_m",
            "dis_gps_hat_mean_m", "dis_gps_hat_max_m",
            "dis_das_only_err_mean_m", "dis_cam_only_err_mean_m",
            "dis_das_worse_than_cam",
            # ── DAS Sensor ──────────────────────────────────────────────────
            "das_n_measurements", "das_snr_mean", "das_snr_min", "das_snr_max",
            "das_snr_std", "das_snr_low_frac",
            "das_sigma_mean_m", "das_sigma_max_m",
            "das_confidence_mean", "das_sigma_v_mean_mps",
            "das_W_est_std_kg", "das_W_est_cv", "das_W_est_kg",
            "das_accept_frac", "das_skip_frac",
            # ── Camera Sensor ────────────────────────────────────────────────
            "cam_n_measurements", "cam_confidence_mean", "cam_confidence_min",
            "cam_confidence_std", "cam_low_conf_frac",
            "cam_sigma_mean_m", "cam_accept_frac",
            # ── GPS Sensor ───────────────────────────────────────────────────
            "gps_n_measurements", "gps_sigma_mean_m",
            # ── Audit ────────────────────────────────────────────────────────
            "aud_skip_total", "aud_skip_frac",
        ]

        # 1. Sort all model features into category order; unknowns go to end.
        ordered_feats = [f for f in _CATEGORY_ORDER if f in feat_cols]
        ordered_feats += [f for f in feat_cols if f not in ordered_feats]
        ordered_idx   = [feat_cols.index(f) for f in ordered_feats]

        # 2. Top-50: keep only the 50 highest-scoring features by max |z| across
        #    all vehicles in this simulation run.  Category ordering is preserved
        #    within the top-50 so related features still appear together.
        Xz_ordered = Xz[:, ordered_idx]
        peak_z     = np.abs(Xz_ordered).max(axis=0)     # max |z| per feature
        _TOP_N     = 30
        n_avail    = Xz_ordered.shape[1]
        if n_avail <= _TOP_N:
            sig_idx = np.arange(n_avail)
        else:
            top_pos = np.argsort(peak_z)[-_TOP_N:]       # top-30 (unordered)
            sig_idx = np.sort(top_pos)                    # restore category order
        show_feats = [ordered_feats[i] for i in sig_idx]
        Z_show     = Xz_ordered[:, sig_idx]   # full values (used for border logic)
        # Cache the top-30 selection so the Excel export can mirror it exactly
        self._ai_top50_feat_names = show_feats
        self._ai_top50_feat_idx   = [feat_cols.index(f) for f in show_feats]

        def _short(f):
            return (f.replace("iv_",  "").replace("kin_", "")
                     .replace("kf_",  "kf ").replace("cov_", "cov ")
                     .replace("dis_", "dis ").replace("das_", "das ")
                     .replace("cam_", "cam ").replace("gps_", "gps ")
                     .replace("aud_", "aud ").replace("sc_",  "sc ")
                     .replace("_mps2", " m/s²").replace("_mps", " m/s")
                     .replace("_m",    " m").replace("_rad", " rad")
                     .replace("_per_s", "/s").replace("_frac", " frac")
                     .replace("_s",    " s").replace("_",     " "))

        ylabels = [_short(f) for f in show_feats]
        n_show  = len(show_feats)
        # Adjust font size to fit rows
        ylabel_fs = max(4, min(8, int(220 / max(n_show, 1))))

        # Color scale: cap at the anomaly threshold so the gradient is meaningful.
        # Cells ≥ threshold saturate to max red/blue — they're anomalous regardless
        # of whether they're 16σ or 160σ. This keeps normal/borderline cells readable.
        heat_lim = float(max(threshold, 1.0))
        Z_display = np.clip(Z_show, -heat_lim, heat_lim)  # clipped only for colour

        ax = fig.add_subplot(111)
        im = ax.imshow(
            Z_display.T, aspect="auto", cmap="RdBu_r", vmin=-heat_lim, vmax=heat_lim,
            interpolation="nearest"
        )
        ax.set_xticks(range(len(track_ids)))
        ax.set_xticklabels(track_ids, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(ylabels, fontsize=ylabel_fs)
        ax.set_title(
            f"Feature Z-Score Heatmap  (threshold = {threshold:.1f}σ)  —  Top {n_show} highest-scoring features"
            f"  ({len(feat_cols)} total in model)",
            fontsize=8, pad=8
        )
        ax.set_xlabel("Vehicle Track", fontsize=9)

        # Thin white separators where the feature family prefix changes
        prev_prefix = None
        for yi, fname in enumerate(show_feats):
            prefix = fname.split("_")[0]
            if prev_prefix is not None and prefix != prev_prefix and yi > 0:
                ax.axhline(yi - 0.5, color="white", linewidth=1.5, zorder=10)
            prev_prefix = prefix

        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.08)
        cb.set_label("Z-Score (σ)  [saturates at threshold]", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        # Black border around cells that exceed the anomaly threshold
        import matplotlib.patches as _mpatch2
        for xi in range(len(track_ids)):
            for yi in range(n_show):
                if abs(Z_show[xi, yi]) > threshold:
                    ax.add_patch(_mpatch2.Rectangle(
                        (xi - 0.5, yi - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.5, zorder=5
                    ))
        fig.tight_layout(pad=1.5)
        self._ai_canvas_heat.draw()

        # ─── Chart B: Score ranking bars ──────────────────────────────────────
        fig2 = self._ai_fig_bars
        fig2.clear()
        sorted_rows = sorted(rows, key=lambda r: r.get("z_score_max", 0), reverse=True)
        vids_s  = [r["track_id"] for r in sorted_rows]
        z_vals  = [r.get("z_score_max", 0) for r in sorted_rows]
        if_vals = [r.get("if_score", 0) for r in sorted_rows]
        zcolors = ["#cc2200" if z > threshold else ("#cc7700" if z > threshold * 0.75 else "#226600") for z in z_vals]

        ax2a = fig2.add_subplot(111)
        bars = ax2a.bar(vids_s, z_vals, color=zcolors, edgecolor="white", linewidth=0.5)
        ax2a.axhline(threshold, color="#cc2200", linewidth=1.5, linestyle="--",
                     label=f"Threshold {threshold:.2f}σ")
        ax2a.axhline(threshold * 0.75, color="#cc7700", linewidth=1.0, linestyle=":",
                     label="Marginal zone")
        for bar, val in zip(bars, z_vals):
            ax2a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                      f"{val:.2f}σ", ha="center", va="bottom", fontsize=8,
                      color="#cc2200" if val > threshold else "#555555")
        ax2a.set_title("Z-Score Anomaly Ranking", fontsize=9)
        ax2a.set_xlabel("Vehicle", fontsize=8)
        ax2a.set_ylabel("Max |Z-Score| (σ)", fontsize=8)
        ax2a.legend(fontsize=7, loc="upper right")
        ax2a.tick_params(labelsize=8)
        fig2.tight_layout(pad=1.4)
        self._ai_canvas_bars.draw()

        # ─── Chart C: Feature distributions ───────────────────────────────────
        try:
            import pandas as pd
        except ImportError:
            pd = None

        fig3 = self._ai_fig_dist
        fig3.clear()
        df_norm = self._ai_df_normal if pd is not None else None

        # ── Feature metadata: units, axis label, type ─────────────────────────
        # type: "continuous" | "count" | "fraction" | "binary"
        _FEAT_META = {
            # Collision / Overlap
            "iv_collision_detected":            ("binary [0/1]",  "collision flag"),
            "iv_overlap_frac":                  ("fraction",       "fraction of time overlapping"),
            "iv_overlap_duration_s":            ("seconds (s)",    "total overlap duration"),
            "iv_collision_risk_proxy":           ("fraction",       "risk proxy score"),
            # Proximity / TTC
            "iv_min_dist_m":                    ("metres (m)",     "minimum gap to nearest vehicle"),
            "iv_mean_min_dist_m":               ("metres (m)",     "mean gap to nearest vehicle"),
            "iv_p10_min_dist_m":                ("metres (m)",     "10th-pctile gap (near misses)"),
            "iv_close_proximity_frac":           ("fraction",       "fraction of time dangerously close"),
            "iv_tailgate_proximity_frac":        ("fraction",       "fraction of time tailgating"),
            "iv_ttc_min_s":                      ("seconds (s)",    "minimum time-to-collision"),
            "iv_ttc_below_2s_frac":             ("fraction",       "fraction of time TTC < 2 s"),
            "iv_closing_speed_max_mps":          ("m/s",           "max closing speed toward vehicle"),
            "iv_rel_speed_at_min_dist_mps":      ("m/s",           "relative speed at closest approach"),
            "iv_decel_at_min_dist_mps2":         ("m/s²",          "deceleration at closest approach"),
            # Stopping / Stall
            "kin_stopped_frac":                  ("fraction",       "fraction of time stopped (v<0.5 m/s)"),
            "kin_stop_event_count":              ("count",          "number of distinct stop events"),
            "kin_max_stopped_steps":             ("count",          "longest stop in timesteps"),
            "kin_max_stopped_duration_s":        ("seconds (s)",    "longest stop duration"),
            # Speed
            "kin_speed_mean_mps":               ("m/s",           "mean speed over run"),
            "kin_speed_max_mps":                ("m/s",           "peak speed"),
            "kin_speed_min_mps":                ("m/s",           "minimum speed"),
            "kin_speed_std_mps":                ("m/s",           "speed variability"),
            "kin_speed_p90_mps":                ("m/s",           "90th-pctile speed"),
            "kin_speed_cv":                      ("ratio",         "speed coeff. of variation (std/mean)"),
            "kin_speed_jump_max_mps":            ("m/s",           "largest single-step speed jump"),
            "kin_speed_over_limit_frac":         ("fraction",       "fraction of time above speed limit"),
            "kin_speed_excess_max_mps":          ("m/s",           "max excess above speed limit"),
            "kin_speed_excess_mean_mps":         ("m/s",           "mean excess above speed limit"),
            # Accel / Jerk
            "kin_accel_mean_abs_mps2":           ("m/s²",          "mean absolute acceleration"),
            "kin_accel_max_abs_mps2":            ("m/s²",          "peak absolute acceleration"),
            "kin_accel_std_mps2":               ("m/s²",          "acceleration variability"),
            "kin_decel_max_mps2":               ("m/s²",          "maximum braking deceleration"),
            "kin_decel_event_count":             ("count",          "number of hard-braking events"),
            "kin_high_decel_frac":              ("fraction",       "fraction of time hard-braking"),
            "kin_jerk_max_mps3":                ("m/s³",          "maximum jerk (rate of accel change)"),
            "kin_jerk_mean_abs_mps3":           ("m/s³",          "mean jerk magnitude"),
            "kin_phys_impossible_v":            ("count",          "timesteps with impossible speed jump"),
            "kin_phys_impossible_a":            ("count",          "timesteps with impossible accel"),
            # Lateral / Heading
            "kin_lateral_vel_mean_mps":          ("m/s",           "mean lateral (sideways) velocity"),
            "kin_lateral_vel_max_mps":           ("m/s",           "peak lateral velocity"),
            "kin_lateral_speed_max_mps":         ("m/s",           "peak lateral speed magnitude"),
            "kin_lateral_accel_max_mps2":        ("m/s²",          "peak lateral acceleration"),
            "kin_lateral_dev_mean_m":            ("metres (m)",    "mean lateral road deviation"),
            "kin_lateral_dev_max_m":             ("metres (m)",    "max lateral road deviation"),
            "kin_lateral_dev_std_m":             ("metres (m)",    "std of lateral deviation"),
            "kin_lateral_peak_to_peak_m":        ("metres (m)",    "lateral oscillation peak-to-peak"),
            "kin_lateral_oscillation_ratio":     ("ratio",         "lateral oscillation / speed ratio"),
            "kin_heading_change_rate_max_rad_per_s": ("rad/s",     "max heading turn rate (⚠ can spike if dt≈0)"),
            "kin_road_heading_diff_mean_rad":    ("radians (rad)", "mean angle vs road direction"),
            # Speed vs Traffic
            "iv_speed_excess_over_others_mps":   ("m/s",           "speed in excess of surrounding vehicles"),
            "iv_speed_ratio_to_others":          ("ratio",         "vehicle speed / mean traffic speed"),
            "iv_speed_pearson_r_traffic":        ("correlation",   "speed correlation with traffic flow"),
            "iv_speed_proximity_ratio_max":      ("ratio",         "max speed-to-proximity ratio"),
            "iv_time_headway_mean_s":            ("seconds (s)",   "mean time gap to vehicle ahead"),
            "iv_n_other_vehicles":               ("count",         "number of surrounding vehicles"),
            "iv_others_mean_speed_when_stopped": ("m/s",           "traffic mean speed when this veh stops"),
            "sc_speed_limit_mps":               ("m/s",           "applicable road speed limit"),
            # Kalman Filter
            "kf_pos_err_mean_m":                ("metres (m)",    "mean Kalman position error"),
            "kf_pos_err_max_m":                 ("metres (m)",    "peak Kalman position error"),
            "kf_pos_err_std_m":                 ("metres (m)",    "variability of Kalman position error"),
            "kf_pos_err_p90_m":                 ("metres (m)",    "90th-pctile Kalman error"),
            "kf_track_rmse_m":                  ("metres (m)",    "Kalman track RMSE"),
            "kf_rolling_rmse_spike_count":       ("count",         "number of rolling RMSE spikes"),
            "kf_sigma_pos_mean_m":              ("metres (m)",    "mean Kalman position uncertainty"),
            "kf_sigma_pos_max_m":               ("metres (m)",    "peak Kalman position uncertainty"),
            "kf_consistency_ratio_mean":         ("ratio",         "innovation / expected (should be ≈1)"),
            "kf_consistency_ratio_max":          ("ratio",         "peak inconsistency ratio"),
            "kf_overconfident_frac":             ("fraction",      "fraction of time filter overconfident"),
            "kf_sigma_growth_rate_during_gaps_m_per_s": ("m/s",   "uncertainty growth rate during dropouts"),
            "kf_post_dropout_jump_m":           ("metres (m)",    "position jump on sensor recovery"),
            # Coverage
            "cov_total_rows":                   ("count",         "total data rows in track"),
            "cov_pred_only_count":              ("count",         "rows with prediction-only (no sensor)"),
            "cov_pred_only_frac":               ("fraction",      "fraction of prediction-only rows"),
            "cov_max_consec_pred":              ("count",         "longest consecutive prediction gap"),
            "cov_n_dropout_events":             ("count",         "number of sensor dropout events"),
            "cov_duration_s":                   ("seconds (s)",   "total track duration"),
            "cov_das_active_frac":              ("fraction",      "fraction of time DAS sensor active"),
            "cov_cam_active_frac":              ("fraction",      "fraction of time Camera active"),
            "cov_gps_active_frac":              ("fraction",      "fraction of time GPS active"),
            "cov_multi_sensor_frac":            ("fraction",      "fraction of time 2+ sensors active"),
            "cov_single_sensor_frac":           ("fraction",      "fraction of time exactly 1 sensor"),
            "cov_no_sensor_frac":               ("fraction",      "fraction of time no sensor active"),
            # Sensor Disagreement
            "dis_das_cam_mean_m":               ("metres (m)",    "mean DAS vs Camera position diff"),
            "dis_das_cam_max_m":                ("metres (m)",    "peak DAS vs Camera disagreement"),
            "dis_das_hat_mean_m":               ("metres (m)",    "mean DAS vs Kalman estimate diff"),
            "dis_das_hat_max_m":                ("metres (m)",    "peak DAS vs Kalman disagreement"),
            "dis_cam_hat_mean_m":               ("metres (m)",    "mean Camera vs Kalman diff"),
            "dis_cam_hat_max_m":                ("metres (m)",    "peak Camera vs Kalman disagreement"),
            "dis_gps_hat_mean_m":               ("metres (m)",    "mean GPS vs Kalman diff"),
            "dis_gps_hat_max_m":                ("metres (m)",    "peak GPS vs Kalman disagreement"),
            # DAS Sensor
            "das_n_measurements":               ("count",         "number of DAS detections"),
            "das_snr_mean":                     ("dB",            "mean DAS signal-to-noise ratio"),
            "das_snr_min":                      ("dB",            "minimum DAS SNR"),
            "das_snr_max":                      ("dB",            "maximum DAS SNR"),
            "das_snr_std":                      ("dB",            "DAS SNR variability"),
            "das_snr_low_frac":                 ("fraction",      "fraction of time DAS SNR low"),
            "das_sigma_mean_m":                 ("metres (m)",    "mean DAS position uncertainty"),
            "das_sigma_max_m":                  ("metres (m)",    "peak DAS uncertainty"),
            "das_confidence_mean":              ("score [0–1]",   "mean DAS confidence score"),
            "das_sigma_v_mean_mps":             ("m/s",           "mean DAS velocity uncertainty"),
            "das_W_est_std_kg":                 ("kg",            "DAS weight estimate variability"),
            "das_W_est_cv":                     ("ratio",         "DAS weight estimate coeff. of variation"),
            "das_W_est_kg":                     ("kg",            "DAS estimated vehicle weight"),
            "das_accept_frac":                  ("fraction",      "fraction of DAS measurements accepted"),
            "das_skip_frac":                    ("fraction",      "fraction of DAS measurements skipped"),
            # Camera
            "cam_n_measurements":               ("count",         "number of Camera detections"),
            "cam_confidence_mean":              ("score [0–1]",   "mean camera confidence"),
            "cam_confidence_min":               ("score [0–1]",   "minimum camera confidence"),
            "cam_confidence_std":               ("score [0–1]",   "camera confidence variability"),
            "cam_low_conf_frac":                ("fraction",      "fraction of time camera low-confidence"),
            "cam_sigma_mean_m":                 ("metres (m)",    "mean camera position uncertainty"),
            "cam_accept_frac":                  ("fraction",      "fraction of camera measurements accepted"),
            # GPS
            "gps_n_measurements":               ("count",         "number of GPS fixes"),
            "gps_sigma_mean_m":                 ("metres (m)",    "mean GPS position uncertainty"),
            # Audit
            "aud_skip_total":                   ("count",         "total audit skips"),
            "aud_skip_frac":                    ("fraction",      "fraction of audit cycles skipped"),
        }

        # Features that are discrete counts and should use bar chart
        _DISCRETE_FEATS = {
            "kin_stop_event_count", "kin_decel_event_count", "kin_max_stopped_steps",
            "kin_phys_impossible_v", "kin_phys_impossible_a", "kf_rolling_rmse_spike_count",
            "cov_total_rows", "cov_pred_only_count", "cov_max_consec_pred",
            "cov_n_dropout_events", "das_n_measurements", "cam_n_measurements",
            "gps_n_measurements", "iv_n_other_vehicles", "aud_skip_total",
        }

        # Show top 6 features by max abs z-score across vehicles, preserving
        # the category order established above (ordered_feats / ordered_idx).
        mean_abs_z_ordered = np.abs(Xz[:, ordered_idx]).max(axis=0)
        # Pick top 6 by peak z, but retain category ordering among them
        top6_mask  = np.argsort(mean_abs_z_ordered)[-6:]
        top6_in_order = sorted(top6_mask)   # keep category order
        dist_feats = [ordered_feats[i] for i in top6_in_order]
        dist_idx   = [feat_cols.index(f) for f in dist_feats if f in feat_cols]
        n_dist = len(dist_feats)
        ncols = 2
        nrows = (n_dist + 1) // 2

        # Colours per vehicle (consistent across all subplots)
        _vid_colors = {}
        _anomaly_vids = {r["track_id"] for r in rows if r.get("flagged")}
        _red_shades   = ["#cc2200", "#e05500", "#b30000", "#ff4400"]
        _green_shades = ["#226600", "#338800", "#115500", "#44aa00"]
        ri_a, ri_n = 0, 0
        for r in rows:
            vid = r["track_id"]
            if r.get("flagged"):
                _vid_colors[vid] = _red_shades[ri_a % len(_red_shades)]; ri_a += 1
            else:
                _vid_colors[vid] = _green_shades[ri_n % len(_green_shades)]; ri_n += 1

        # Build subplots with reserved space at top for shared legend + title
        fig3.subplots_adjust(top=0.78, hspace=1.1, wspace=0.40)
        axes_dist = []
        for pi, (fi, fname) in enumerate(zip(dist_idx, dist_feats)):
            ax = fig3.add_subplot(nrows, ncols, pi + 1)
            axes_dist.append(ax)
            short = _short(fname)
            meta       = _FEAT_META.get(fname, ("value", fname.replace("_", " ")))
            units_lbl  = meta[0]
            desc_lbl   = meta[1]
            is_discrete = fname in _DISCRETE_FEATS

            mu_n_val = sg_n_val = None
            vis_lo = vis_hi = None
            norm_vals = np.array([])

            if df_norm is not None and fname in df_norm.columns:
                norm_vals = pd.to_numeric(df_norm[fname], errors="coerce").dropna().values

            # ── Back-transform vehicle z-scores to original scale ──────────────
            veh_raw = {}
            for ri, row in enumerate(rows):
                vid   = row["track_id"]
                z_val = float(Xz[ri, fi]) if fi < Xz.shape[1] else 0.0
                if len(norm_vals):
                    if mu_n_val is None:
                        mu_n_val = float(np.mean(norm_vals))
                        sg_n_val = float(np.std(norm_vals)) or 1.0
                    veh_raw[vid] = (mu_n_val + z_val * sg_n_val, z_val)
                else:
                    veh_raw[vid] = (z_val, z_val)

            if is_discrete and len(norm_vals):
                # ── Discrete count feature → bar chart ────────────────────────
                from collections import Counter
                counts = Counter(int(round(v)) for v in norm_vals)
                x_vals = sorted(counts.keys())
                y_vals = [counts[x] / len(norm_vals) for x in x_vals]
                ax.bar(x_vals, y_vals, color="#3a6fbf", alpha=0.5,
                       width=0.6, zorder=2, label="training")
                # Mark each vehicle as a dot + label
                y_max = max(y_vals) if y_vals else 1.0
                for ri, row in enumerate(rows):
                    vid   = row["track_id"]
                    v_raw, z_v = veh_raw.get(vid, (0, 0))
                    col   = _vid_colors.get(vid, "#555555")
                    mk    = "v" if row.get("flagged") else "o"
                    ax.scatter([v_raw], [y_max * 0.85 - ri * y_max * 0.12],
                               color=col, marker=mk, s=28, zorder=6)
                    ax.text(v_raw, y_max * 0.85 - ri * y_max * 0.12,
                            f" {vid}", fontsize=5.5, color=col, va="center")
                ax.set_ylabel("Training frequency", fontsize=6)

            else:
                # ── Continuous feature → histogram with visible-window clamp ──
                if len(norm_vals):
                    if mu_n_val is None:
                        mu_n_val = float(np.mean(norm_vals))
                        sg_n_val = float(np.std(norm_vals)) or 1.0
                    vis_lo = mu_n_val - 4.0 * sg_n_val
                    vis_hi = mu_n_val + 4.0 * sg_n_val
                    for vid, (v_raw, z_v) in veh_raw.items():
                        if abs(z_v) <= 8.0:
                            vis_lo = min(vis_lo, v_raw - 0.5 * sg_n_val)
                            vis_hi = max(vis_hi, v_raw + 0.5 * sg_n_val)
                    span = max(vis_hi - vis_lo, 1e-9)
                    vis_lo -= 0.05 * span; vis_hi += 0.05 * span

                    clip_vals = norm_vals[(norm_vals >= vis_lo) & (norm_vals <= vis_hi)]
                    if len(clip_vals):
                        ax.hist(clip_vals, bins=40, density=True, alpha=0.5,
                                color="#3a6fbf", zorder=2)
                    # Normal zone shading ± 2σ (inner, lighter) and ±4σ (outer, very faint)
                    ax.axvspan(max(vis_lo, mu_n_val - 2.0 * sg_n_val),
                               min(vis_hi, mu_n_val + 2.0 * sg_n_val),
                               alpha=0.13, color="#226600", zorder=1, label="±2σ zone")
                    ax.axvspan(max(vis_lo, mu_n_val - 4.0 * sg_n_val),
                               min(vis_hi, mu_n_val + 4.0 * sg_n_val),
                               alpha=0.05, color="#226600", zorder=1)
                    # Mean marker
                    if vis_lo <= mu_n_val <= vis_hi:
                        ax.axvline(mu_n_val, color="#226600", linewidth=1.2,
                                   linestyle="-", alpha=0.7, zorder=3)
                        ax.text(mu_n_val, 0.97, "μ", transform=ax.get_xaxis_transform(),
                                ha="center", va="top", fontsize=6, color="#226600", alpha=0.8)
                    ax.set_xlim(vis_lo, vis_hi)
                ax.set_ylabel("Prob. density", fontsize=6)

                # Per-vehicle lines: draw within range, annotate at edge if extreme
                ann_y = 0.93
                for ri, row in enumerate(rows):
                    vid   = row["track_id"]
                    v_raw, z_v = veh_raw.get(vid, (0, 0))
                    col   = _vid_colors.get(vid, "#555555")
                    ls    = "--" if row.get("flagged") else ":"
                    if vis_lo is not None and (v_raw < vis_lo or v_raw > vis_hi):
                        is_right = v_raw > vis_hi
                        x_ann = vis_hi if is_right else vis_lo
                        arrow = "→" if is_right else "←"
                        ax.annotate(
                            f"{arrow}{vid}: {z_v:+.0f}σ",
                            xy=(x_ann, 0), xycoords=("data", "axes fraction"),
                            xytext=(x_ann, ann_y),
                            textcoords=("data", "axes fraction"),
                            fontsize=5.5, color=col,
                            ha="right" if is_right else "left",
                            va="top", fontweight="bold",
                        )
                        ann_y -= 0.13
                    else:
                        ax.axvline(v_raw, color=col, linewidth=1.8,
                                   linestyle=ls, zorder=5)

            # ── Axis labels with units ─────────────────────────────────────────
            ax.set_title(f"{short}\n({desc_lbl})", fontsize=7, pad=2)
            ax.set_xlabel(units_lbl, fontsize=6.5)
            ax.tick_params(labelsize=6)

        # ── Shared legend + figure title in the reserved space above subplots ──
        import matplotlib.patches as _mpatch
        import matplotlib.lines  as _mlines
        legend_handles = [
            _mpatch.Patch(color="#3a6fbf", alpha=0.5, label="Training distribution (normal data)"),
            _mpatch.Patch(color="#226600", alpha=0.18, label="±2σ normal zone"),
            _mpatch.Patch(color="#226600", alpha=0.08, label="±4σ outer zone"),
            _mlines.Line2D([], [], color="#226600", linewidth=1.2, linestyle="-",
                           alpha=0.7, label="μ  (training mean)"),
        ]
        for r in rows:
            vid = r["track_id"]
            col = _vid_colors.get(vid, "#555555")
            ls  = "--" if r.get("flagged") else ":"
            lbl = f"{vid}  {'⚠ ANOMALY' if r.get('flagged') else '✓ normal'}"
            legend_handles.append(
                _mlines.Line2D([], [], color=col, linewidth=1.8,
                               linestyle=ls, label=lbl)
            )
        fig3.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            fontsize=7,
            frameon=True,
            framealpha=0.92,
            bbox_to_anchor=(0.5, 0.97),
        )
        fig3.text(
            0.5, 0.865,
            "Top Anomalous Features — Measured Values vs Normal Training Distribution",
            ha="center", va="center", fontsize=9, fontweight="bold",
            transform=fig3.transFigure,
        )
        fig3.text(
            0.5, 0.845,
            "Dashed line = anomalous vehicle  ·  Dotted line = normal vehicle  ·"
            "  Arrow+σ label = value is off-screen (extreme)  ·  μ = training mean",
            ha="center", va="center", fontsize=7, color="#555555",
            transform=fig3.transFigure,
        )
        self._ai_canvas_dist.draw()

        # ── Populate both tables ──────────────────────────────────────────────
        try:
            _catalog: dict = {}
            try:
                import sys as _sys2
                _rr = str(Path(__file__).resolve().parents[3])
                if _rr not in _sys2.path:
                    _sys2.path.insert(0, _rr)
                from anomaly_model.make_features_excel import FEATURE_CATALOG_RICH
                _catalog = FEATURE_CATALOG_RICH
            except Exception:
                pass

            # Table 1 (Feature Distributions tab): measured values per vehicle
            tbl = self._ai_dist_table
            for iid in tbl.get_children():
                tbl.delete(iid)

            for fi, fname in zip(dist_idx, dist_feats):
                meta      = _FEAT_META.get(fname, ("value", fname.replace("_", " ")))
                units_lbl = meta[0]

                mu_t = sg_t = None
                if df_norm is not None and fname in df_norm.columns:
                    nv = pd.to_numeric(df_norm[fname], errors="coerce").dropna().values
                    if len(nv):
                        mu_t = float(np.mean(nv))
                        sg_t = float(np.std(nv)) or 1.0

                for ri, row in enumerate(rows):
                    vid   = row["track_id"]
                    z_val = float(Xz[ri, fi]) if fi < Xz.shape[1] else 0.0
                    if mu_t is not None:
                        v_raw = mu_t + z_val * sg_t
                        rng_s = f"{mu_t - 2*sg_t:.3g} → {mu_t + 2*sg_t:.3g} {units_lbl}"
                        val_s = f"{v_raw:.4g} {units_lbl}"
                    else:
                        rng_s = "n/a"
                        val_s = f"{z_val:.4g}"

                    z_abs = abs(z_val)
                    flagged = row.get("flagged", False)
                    if flagged or z_abs > threshold:
                        tag, verdict = "anomaly", "⚠ ANOMALY"
                    elif z_abs > threshold * 0.75:
                        tag, verdict = "marginal", "marginal"
                    else:
                        tag, verdict = "normal",  "✓ normal"

                    cat_entry2   = _catalog.get(fname, {})
                    anomaly_type = cat_entry2.get("helps_identify") or "—"
                    if anomaly_type == "—":   # metadata/identifier rows — keep dash
                        anomaly_type = "—"
                    tbl.insert("", "end",
                        values=(fname, vid, val_s, f"{z_val:+.2f}σ",
                                rng_s, verdict, anomaly_type),
                        tags=(tag,))

            # Table 2 (left panel): catalog info — one row per feature in the top-30
            # Use self._ai_top50_feat_names which holds exactly the top-30 heatmap features
            cat_tbl = self._ai_catalog_table
            for iid in cat_tbl.get_children():
                cat_tbl.delete(iid)

            top30 = getattr(self, "_ai_top50_feat_names", dist_feats) or dist_feats
            for idx, fname in enumerate(top30):
                meta         = _FEAT_META.get(fname, ("value", fname.replace("_", " ")))
                desc_lbl     = meta[1]
                cat_entry    = _catalog.get(fname, {})
                cat_desc     = cat_entry.get("description") or desc_lbl
                cat_range    = cat_entry.get("output_range") or "—"
                anomaly_type = cat_entry.get("helps_identify") or "—"
                row_tag      = "row_odd" if idx % 2 else "row_even"
                cat_tbl.insert("", "end",
                    values=(fname, anomaly_type, cat_desc, cat_range),
                    tags=(row_tag,))
        except Exception:
            pass  # tables are best-effort, never block the rest of the render

        # ─── Chart D: CUSUM temporal anomaly curves ────────────────────────────
        # Redesigned to show WHEN an anomaly started, using two parallel signals:
        #
        #  1. Speed-drop CUSUM (lower):  S_dn[t] = max(0, S_dn[t-1] + (baseline – v[t] – K_v))
        #     Fires when vehicle speed falls persistently below its own pre-anomaly baseline.
        #     K_v = slack in m/s; h_v = alarm threshold.
        #
        #  2. Proximity CUSUM (upper):   S_pr[t] = max(0, S_pr[t-1] + (d_normal – dist[t] – K_d))
        #     Fires when inter-vehicle distance falls persistently below its own pre-anomaly normal.
        #     Only shown when proximity time-series is available.
        #
        # The composite CUSUM = max(S_dn, S_pr) at each timestep.
        # A vertical red line marks the first moment the composite exceeds the alarm threshold.
        #
        # Parameters tuned for the simulator's typical speed scale (5–30 m/s):
        CUSUM_K_V  = 2.0    # speed slack m/s — ignore small normal fluctuations
        CUSUM_K_D  = 3.0    # distance slack m — ignore gradual approach
        CUSUM_H    = 12.0   # alarm threshold for both signals
        BASELINE_FRAC = 0.2 # fraction of run used as pre-anomaly reference baseline

        if self._ai_fig_cusum is None:
            return
        fig4 = self._ai_fig_cusum
        fig4.clear()

        # ── Redesigned temporal view ───────────────────────────────────────────
        # One panel per vehicle.  Shows speed over time with:
        #   • Green band  = "normal" speed zone (vehicle's own first-20% baseline ± 2σ)
        #   • Speed line  = coloured by vehicle, red segments where speed alarm fires
        #   • Red fill    = periods where the CUSUM statistic has crossed its alarm
        #                   threshold (i.e. sustained deviation from normal)
        #   • ⚠ marker    = earliest moment the alarm first fires
        #   • Proximity   = dotted grey line on right y-axis when available
        # This directly answers "when did this vehicle start behaving abnormally?"

        n_vehs     = len(rows)
        n_cols     = min(2, n_vehs)
        n_rows_fig = (n_vehs + n_cols - 1) // n_cols
        fig4.subplots_adjust(hspace=0.65, wspace=0.40,
                             top=0.89, bottom=0.08, left=0.10, right=0.97)
        fig4.text(0.5, 0.97,
                  "Temporal Behaviour — Speed Trace & Anomaly Onset",
                  ha="center", va="top", fontsize=9, fontweight="bold",
                  transform=fig4.transFigure)
        fig4.text(0.5, 0.93,
                  "Green band = normal speed range.  "
                  "Red fill = sustained anomaly (CUSUM alarm).  "
                  "⚠ = first alarm onset.",
                  ha="center", va="top", fontsize=7, color="#555555",
                  transform=fig4.transFigure)

        def _pages_cusum_lower(values, baseline, K, h):
            """Page's CUSUM: accumulates deficit below (baseline − K); resets on recovery."""
            S = np.zeros(len(values))
            s = 0.0
            for i, v in enumerate(values):
                s = max(0.0, s + (baseline - v - K))
                S[i] = s
            return S

        for pi, row in enumerate(rows):
            vid    = row["track_id"]
            col    = _vid_colors.get(vid, "#555555")
            flagged = row.get("flagged", False)
            _ts    = timeseries.get(vid, ())
            times_v  = _ts[0] if len(_ts) > 0 else np.array([])
            speeds_v = _ts[1] if len(_ts) > 1 else np.array([])
            dists_t  = _ts[2] if len(_ts) > 2 else np.array([])
            dists_v  = _ts[3] if len(_ts) > 3 else np.array([])

            r_idx = pi // n_cols
            c_idx = pi %  n_cols
            ax = fig4.add_subplot(n_rows_fig, n_cols,
                                  r_idx * n_cols + c_idx + 1)

            if len(times_v) < 4:
                ax.text(0.5, 0.5, "no time-series data\n(live mode only)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#888")
                status = "⚠ ANOMALY" if flagged else "✓ normal"
                ax.set_title(f"{vid}  {status}", fontsize=8,
                             color="#cc2200" if flagged else "#226600")
                continue

            # ── Baseline from first 20 % of run ───────────────────────────────
            n_base = max(4, int(len(speeds_v) * BASELINE_FRAC))
            v_base = float(np.mean(speeds_v[:n_base]))
            v_std  = float(np.std(speeds_v[:n_base])) or (v_base * 0.15 + 0.5)

            # ── CUSUM on speed-drop ────────────────────────────────────────────
            S_speed = _pages_cusum_lower(speeds_v, v_base, CUSUM_K_V, CUSUM_H)

            # Optional proximity CUSUM (interpolated to speed timeline)
            has_dists   = len(dists_v) >= 3
            S_composite = S_speed.copy()
            if has_dists:
                n_base_d = max(2, int(len(dists_v) * BASELINE_FRAC))
                d_base   = float(np.mean(dists_v[:n_base_d]))
                S_prox   = _pages_cusum_lower(dists_v, d_base, CUSUM_K_D, CUSUM_H)
                S_prox_i = np.interp(times_v, dists_t, S_prox,
                                     left=0.0, right=S_prox[-1])
                S_composite = np.maximum(S_speed, S_prox_i)

            alarm_mask = S_composite > CUSUM_H

            # ── Green "normal" band (baseline ± 2σ, but at least ±1 m/s) ─────
            band_lo = max(0.0, v_base - max(2.0 * v_std, 1.0))
            band_hi = v_base + max(2.0 * v_std, 1.0)
            ax.axhspan(band_lo, band_hi, alpha=0.18, color="#226600", zorder=1,
                       label="normal zone")

            # ── Speed line ───────────────────────────────────────────────────
            ax.plot(times_v, speeds_v, color=col, linewidth=1.3,
                    alpha=0.9, zorder=3)
            ax.axhline(v_base, color=col, linewidth=0.8, linestyle="--",
                       alpha=0.45)

            # ── Red fill for CUSUM alarm periods ──────────────────────────────
            if alarm_mask.any():
                ax.fill_between(times_v, 0, np.where(alarm_mask, speeds_v, 0),
                                where=alarm_mask,
                                color="#cc2200", alpha=0.22, zorder=2,
                                label="alarm period")
                onset_idx = int(np.argmax(alarm_mask))
                t_onset   = float(times_v[onset_idx])
                ax.axvline(t_onset, color="#cc2200", linewidth=1.4,
                           linestyle=":", alpha=0.9, zorder=4)
                ax.text(t_onset, band_hi * 1.05,
                        f" ⚠ t={t_onset:.1f}s",
                        fontsize=6, color="#cc2200", va="bottom", zorder=5)
                verdict_color = "#cc2200"
                verdict_txt   = f"⚠ alarm at {t_onset:.1f} s"
            else:
                verdict_color = "#226600" if not flagged else "#e05500"
                verdict_txt   = "✓ no temporal alarm" if not flagged else "marginal"

            # ── Proximity on twin axis ─────────────────────────────────────────
            if has_dists:
                ax_d = ax.twinx()
                ax_d.plot(dists_t, dists_v, color="#888888", linewidth=0.8,
                          linestyle=":", alpha=0.55)
                ax_d.set_ylabel("nearest (m)", fontsize=5, color="#888888")
                ax_d.tick_params(axis="y", labelsize=5, labelcolor="#888888")

            ax.set_title(f"{vid}  —  {verdict_txt}", fontsize=7,
                         color=verdict_color, pad=2)
            ax.set_xlabel("Time (s)", fontsize=6)
            ax.set_ylabel("Speed (m/s)", fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_ylim(bottom=0)

            # ── Per-panel legend ──────────────────────────────────────────────
            import matplotlib.patches as _mpatch3
            import matplotlib.lines   as _mlines3
            _leg_handles = [
                _mpatch3.Patch(color="#226600", alpha=0.25,
                               label="normal zone (baseline ± 2σ)"),
            ]
            if alarm_mask.any():
                _leg_handles.append(
                    _mpatch3.Patch(color="#cc2200", alpha=0.28,
                                   label="CUSUM alarm — sustained deviation"))
            _leg_handles.append(
                _mlines3.Line2D([], [], color=col, linewidth=1.3,
                                label="speed (m/s)"))
            if has_dists:
                _leg_handles.append(
                    _mlines3.Line2D([], [], color="#888888", linewidth=0.8,
                                    linestyle=":", label="nearest vehicle (m)"))
            ax.legend(handles=_leg_handles, fontsize=4.5,
                      loc="upper right", framealpha=0.75)

        self._ai_canvas_cusum.draw()

    # ──────────────────────────────────────────────────────────────────────────
    def _on_score_row_selected(self, _event=None):
        """Populate the Feature Inspector when a vehicle row is clicked."""
        try:
            import numpy as np
            result = getattr(self, "_ai_analysis_result", None)
            if not result:
                return
            sel = self.tv_ai_scores.selection()
            if not sel:
                return

            # Read vehicle id from the selected row (first column value)
            row_vals = self.tv_ai_scores.item(sel[0], "values")
            if not row_vals:
                return
            vid = str(row_vals[0])

            rows       = result.get("rows", [])
            feat_cols  = result.get("feat_cols", [])
            z_matrix   = result.get("z_matrix")
            threshold  = result.get("threshold", 15.0)

            # Match row index for this vehicle
            row_idx = next((i for i, r in enumerate(rows)
                            if str(r.get("track_id", "")) == vid), None)
            if row_idx is None or z_matrix is None:
                return

            z_row = z_matrix[row_idx]  # shape: (n_features,)

            # --- Feature metadata (units) reused from _populate_anomaly_intel ---
            _UNITS = {
                "iv_collision_detected": "binary",
                "iv_collision_count": "count",
                "iv_min_dist_m": "m",
                "iv_time_to_collision_min_s": "s",
                "iv_ttc_below_2s_frac": "fraction",
                "iv_closing_speed_max_mps": "m/s",
                "iv_rel_speed_at_min_dist_mps": "m/s",
                "iv_decel_at_min_dist_mps2": "m/s²",
                "kin_stopped_frac": "fraction",
                "kin_stop_event_count": "count",
                "kin_max_stopped_steps": "count",
                "kin_max_stopped_duration_s": "s",
                "kin_speed_mean_mps": "m/s",
                "kin_speed_max_mps": "m/s",
                "kin_speed_min_mps": "m/s",
                "kin_speed_std_mps": "m/s",
                "kin_speed_p90_mps": "m/s",
                "kin_speed_cv": "ratio",
                "kin_speed_jump_max_mps": "m/s",
                "kin_speed_over_limit_frac": "fraction",
                "kin_speed_excess_max_mps": "m/s",
                "kin_speed_excess_mean_mps": "m/s",
                "kin_accel_mean_abs_mps2": "m/s²",
                "kin_accel_max_abs_mps2": "m/s²",
                "kin_accel_std_mps2": "m/s²",
                "kin_decel_max_mps2": "m/s²",
                "kin_decel_event_count": "count",
                "kin_high_decel_frac": "fraction",
                "kin_jerk_max_mps3": "m/s³",
                "kin_jerk_mean_abs_mps3": "m/s³",
                "kin_phys_impossible_v": "count",
                "kin_phys_impossible_a": "count",
                "kin_lateral_speed_max_mps": "m/s",
                "kin_lateral_accel_max_mps2": "m/s²",
                "kin_heading_change_rate_max_rad_per_s": "rad/s",
                "kf_pos_err_mean_m": "m",
                "kf_pos_err_max_m": "m",
                "kf_track_rmse_m": "m",
                "kf_consistency_ratio_mean": "ratio",
                "kf_consistency_ratio_max": "ratio",
                "kf_overconfident_frac": "fraction",
                "cov_pred_only_frac": "fraction",
                "cov_n_dropout_events": "count",
                "cov_duration_s": "s",
                "cov_das_active_frac": "fraction",
                "cov_cam_active_frac": "fraction",
                "cov_gps_active_frac": "fraction",
                "dis_das_cam_mean_m": "m",
                "dis_das_cam_max_m": "m",
                "das_snr_mean": "dB",
                "das_snr_min": "dB",
                "das_snr_std": "dB",
                "das_snr_low_frac": "fraction",
                "das_sigma_mean_m": "m",
                "das_confidence_mean": "score",
                "cam_confidence_mean": "score",
                "cam_confidence_min": "score",
                "cam_low_conf_frac": "fraction",
            }

            tbl = self._ai_inspector
            for iid in tbl.get_children():
                tbl.delete(iid)

            # Header
            tbl.insert("", "end",
                values=(f"Vehicle: {vid}", "Value", "Z-score", "Status"),
                tags=("header",))

            # Sort features: worst z-score first
            feat_order = sorted(range(len(feat_cols)),
                                key=lambda i: abs(float(z_row[i])) if i < len(z_row) else 0,
                                reverse=True)

            for fi in feat_order:
                fname  = feat_cols[fi]
                z_val  = float(z_row[fi]) if fi < len(z_row) else 0.0
                units  = _UNITS.get(fname, "")
                # Back-calculate raw value from z-score using model scaler
                model_data = getattr(self, "_ai_model_data", None) or {}
                scaler_mean = model_data.get("scaler_mean", [])
                scaler_std  = model_data.get("scaler_std",  [])
                if fi < len(scaler_mean) and fi < len(scaler_std):
                    raw = float(scaler_mean[fi]) + z_val * float(scaler_std[fi])
                    val_str = f"{raw:.4g} {units}".strip()
                else:
                    val_str = f"z={z_val:+.3f}"

                z_abs = abs(z_val)
                if z_abs > threshold:
                    tag, verdict = "anomaly", "⚠ ANOMALY"
                elif z_abs > threshold * 0.75:
                    tag, verdict = "marginal", "marginal"
                else:
                    tag, verdict = "normal",  "✓ ok"

                short = fname.replace("kin_", "").replace("kf_", "kf:") \
                             .replace("cov_", "cov:").replace("iv_", "iv:") \
                             .replace("das_", "das:").replace("cam_", "cam:") \
                             .replace("dis_", "dis:").replace("gps_", "gps:") \
                             .replace("sc_", "sc:").replace("aud_", "aud:")
                tbl.insert("", "end",
                    values=(short, val_str, f"{z_val:+.2f}σ", verdict),
                    tags=(tag,))
        except Exception:
            pass  # inspector is best-effort

    # ──────────────────────────────────────────────────────────────────────────
    def _run_cusum_demo(self):
        """Load synthetic vehicle trajectories so the CUSUM tab has visible data.

        Creates 4 vehicles:
          DEMO_A  — normal: steady cruise at ~14 m/s
          DEMO_B  — normal: steady cruise at ~11 m/s
          DEMO_C  — anomaly: sudden hard stop at t = 20 s (collision)
          DEMO_D  — anomaly: gradual speed bleed from t = 15 s (stall / jam)

        The synthetic feature z-scores are calibrated so that DEMO_C and DEMO_D
        are flagged far above the anomaly threshold, and the CUSUM alarm fires
        clearly for both.
        """
        import numpy as np

        rng = np.random.default_rng(42)
        n   = 300                            # 300 time-steps over 0–60 s
        t   = np.linspace(0.0, 60.0, n)
        dt  = t[1] - t[0]

        def _noisy(base, sigma=0.4):
            return base + rng.normal(0, sigma, n)

        # ── Speed traces ──────────────────────────────────────────────────────
        s_A = _noisy(14.0, 0.3)                                  # steady normal
        s_B = _noisy(11.0, 0.3)                                  # steady normal

        # DEMO_C: normal for 20 s, then hard collision stop (v → 0 in ~3 s)
        s_C = np.where(t < 20.0,
                       _noisy(14.0, 0.3),
                       np.maximum(0.0, 14.0 - 5.0 * (t - 20.0)) + rng.normal(0, 0.2, n))

        # DEMO_D: normal for 15 s, then gradual bleed to near-stop (traffic jam)
        s_D = np.where(t < 15.0,
                       _noisy(13.0, 0.3),
                       np.maximum(0.5, 13.0 - 0.85 * (t - 15.0)) + rng.normal(0, 0.3, n))

        # ── Inter-vehicle proximity (sampled every 5 steps) ───────────────────
        ts = t[::5]
        n_s = len(ts)

        def _flat_dist(base, sigma=1.5):
            return base + rng.normal(0, sigma, n_s)

        # DEMO_C proximity: collapses to near-zero at t=20 (crash)
        d_C = np.where(ts < 20.0,
                       _flat_dist(28.0),
                       np.maximum(0.3, 28.0 - 5.5 * (ts - 20.0)) + rng.normal(0, 0.4, n_s))

        # DEMO_D proximity: narrows steadily as jam forms
        d_D = np.where(ts < 15.0,
                       _flat_dist(32.0),
                       np.maximum(1.0, 32.0 - 1.8 * (ts - 15.0)) + rng.normal(0, 0.5, n_s))

        timeseries = {
            "DEMO_A": (t, s_A, ts, _flat_dist(30.0)),
            "DEMO_B": (t, s_B, ts, _flat_dist(27.0)),
            "DEMO_C": (t, s_C, ts, d_C),
            "DEMO_D": (t, s_D, ts, d_D),
        }

        # ── Synthetic feature z-scores ────────────────────────────────────────
        # Use the live model's feature list if available; otherwise a stub list.
        model_data  = getattr(self, "_ai_model_data", None) or {}
        feat_cols   = model_data.get("feature_cols", [
            "kin_stopped_frac", "kin_stop_event_count",
            "kin_speed_mean_mps", "kin_speed_min_mps",
            "kin_decel_max_mps2", "iv_min_dist_m",
            "iv_collision_detected", "iv_closing_speed_max_mps",
        ])
        threshold   = float(model_data.get("threshold", 15.19))
        n_feat      = len(feat_cols)

        # Build a (4, n_feat) z-score matrix
        # Vehicles A & B: all z-scores small (normal)
        Xz = rng.normal(0, 1.0, (4, n_feat))

        # DEMO_C: spike several stop/collision features way above threshold
        for fname, z_val in [
            ("kin_stopped_frac",        threshold * 3.5),
            ("kin_stop_event_count",    threshold * 2.8),
            ("kin_speed_min_mps",      -threshold * 2.1),
            ("kin_decel_max_mps2",      threshold * 4.2),
            ("iv_collision_detected",   threshold * 6.0),
            ("iv_min_dist_m",          -threshold * 3.0),
            ("iv_closing_speed_max_mps",threshold * 3.5),
        ]:
            if fname in feat_cols:
                Xz[2, feat_cols.index(fname)] = z_val

        # DEMO_D: spike stall/jam features
        for fname, z_val in [
            ("kin_stopped_frac",        threshold * 2.0),
            ("kin_stop_event_count",    threshold * 1.8),
            ("kin_speed_mean_mps",     -threshold * 1.5),
            ("kin_speed_min_mps",      -threshold * 2.3),
            ("iv_min_dist_m",          -threshold * 1.6),
        ]:
            if fname in feat_cols:
                Xz[3, feat_cols.index(fname)] = z_val

        import numpy as _np
        z_abs_max  = _np.abs(Xz).max(axis=1)
        z_feat_idx = _np.abs(Xz).argmax(axis=1)

        rows = []
        for i, vid in enumerate(["DEMO_A", "DEMO_B", "DEMO_C", "DEMO_D"]):
            fi    = int(z_feat_idx[i])
            rows.append({
                "track_id":    vid,
                "z_score_max": float(z_abs_max[i]),
                "if_score":    0.0,
                "flagged":     bool(z_abs_max[i] > threshold),
                "top_feature": feat_cols[fi] if fi < len(feat_cols) else "?",
            })

        result = {
            "rows":             rows,
            "feat_cols":        feat_cols,
            "z_matrix":         Xz,
            "threshold":        threshold,
            "n_features_live":  n_feat,
            "n_features_model": n_feat,
            "timeseries":       timeseries,
        }

        self._ai_status.set(
            "CUSUM Demo loaded — 4 synthetic vehicles (2 normal, 2 anomalous). "
            "Switch to the CUSUM tab to see the speed traces and alarm onset markers."
        )
        self._populate_anomaly_intel(result)

    # ──────────────────────────────────────────────────────────────────────────
    def _ai_export_report(self):
        """Export Anomaly Intel as a formatted Excel workbook + embedded charts."""
        result = getattr(self, "_ai_analysis_result", None)
        if not result or not result.get("rows"):
            from tkinter import messagebox
            messagebox.showinfo(
                "Export Anomaly Intel",
                "No analysis results yet.\nClick 'Analyze Simulation' first."
            )
            return

        try:
            from openpyxl import Workbook
            from openpyxl.styles import (Font, PatternFill, Alignment,
                                          Border, Side, GradientFill)
            from openpyxl.utils import get_column_letter
            from openpyxl.formatting.rule import ColorScaleRule
            from openpyxl.drawing.image import Image as XLImage
            import numpy as np, datetime, io, os
        except ImportError as e:
            from tkinter import messagebox
            messagebox.showerror("Export Error", f"Missing library: {e}")
            return

        from tkinter import filedialog, messagebox
        scenario = getattr(self, "_current_scenario", None) or "anomaly_report"
        if hasattr(scenario, "name"):
            scenario = scenario.name
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        xl_path = filedialog.asksaveasfilename(
            title="Save Anomaly Intel Report",
            initialfile=f"{scenario}_{ts}.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
        )
        if not xl_path:
            return

        rows      = result["rows"]
        feat_cols = result["feat_cols"]
        Xz        = result["z_matrix"]
        threshold = result["threshold"]
        n_veh     = len(rows)
        n_feat    = len(feat_cols)
        n_flagged = sum(1 for r in rows if r.get("flagged"))

        # ── Style helpers ─────────────────────────────────────────────────────
        def _argb(c):
            """Ensure 8-char aRGB string (openpyxl requirement)."""
            return ("FF" + c) if len(c) == 6 else c

        def _hdr_fill(hex_color):
            return PatternFill("solid", fgColor=_argb(hex_color))

        def _font(bold=False, color="000000", size=10, name="Arial"):
            return Font(bold=bold, color=_argb(color), size=size, name=name)

        def _align(h="center", v="center", wrap=False):
            return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

        def _border(style="thin"):
            s = Side(style=style, color=_argb("BBBBBB"))
            return Border(left=s, right=s, top=s, bottom=s)

        def _thick_bottom():
            thin = Side(style="thin",   color=_argb("BBBBBB"))
            thk  = Side(style="medium", color=_argb("2C4F7C"))
            return Border(left=thin, right=thin, top=thin, bottom=thk)

        NAVY   = "1F3864"
        RED_BG = "FFCCCC"; RED_TXT = "C00000"
        GRN_BG = "E2EFDA"; GRN_TXT = "375623"
        ALT_BG = "EEF3FF"; WHT_BG  = "FFFFFF"

        try:
            wb = Workbook()

            # ══════════════════════════════════════════════════════════════
            # SHEET 1 — SUMMARY
            # ══════════════════════════════════════════════════════════════
            ws1 = wb.active
            ws1.title = "Summary"
            ws1.sheet_view.showGridLines = False

            # Banner
            verdict_txt = (f"⚠  ANOMALY DETECTED — {n_flagged} of {n_veh} vehicles flagged"
                           if n_flagged else
                           f"✓  All {n_veh} vehicles within normal range")
            ws1.merge_cells("A1:G1")
            c = ws1["A1"]
            c.value = f"Anomaly Intel Report  |  {scenario}"
            c.font  = _font(bold=True, color="FFFFFF", size=14)
            c.fill  = _hdr_fill(NAVY)
            c.alignment = _align()
            ws1.row_dimensions[1].height = 28

            ws1.merge_cells("A2:G2")
            c = ws1["A2"]
            c.value = verdict_txt
            c.font  = _font(bold=True,
                            color=RED_TXT if n_flagged else GRN_TXT, size=11)
            c.fill  = _hdr_fill(RED_BG if n_flagged else GRN_BG)
            c.alignment = _align()
            ws1.row_dimensions[2].height = 22

            ws1.merge_cells("A3:G3")
            ws1["A3"].value = (
                f"Model: {n_feat} features  ·  Threshold: {threshold:.2f}σ  ·  "
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            ws1["A3"].font      = _font(color="555555", size=9)
            ws1["A3"].fill      = _hdr_fill("F2F2F2")
            ws1["A3"].alignment = _align()
            ws1.row_dimensions[3].height = 16

            # Column headers (row 5)
            hdrs = ["Vehicle", "Max |Z-Score|", "Status",
                    "Peak Feature", "Z at Peak", "# Features Computed", "Flagged?"]
            col_widths_s = [14, 16, 12, 40, 12, 22, 10]
            for ci, (h, w) in enumerate(zip(hdrs, col_widths_s), 1):
                cell = ws1.cell(row=5, column=ci, value=h)
                cell.font      = _font(bold=True, color="FFFFFF", size=10)
                cell.fill      = _hdr_fill("2C4F7C")
                cell.alignment = _align()
                cell.border    = _thick_bottom()
                ws1.column_dimensions[get_column_letter(ci)].width = w
            ws1.row_dimensions[5].height = 20

            # Data rows
            n_live = result.get("n_features_live", "?")
            for ri, row in enumerate(rows):
                r_xl = ri + 6
                flagged   = row.get("flagged", False)
                bg        = RED_BG if flagged else (ALT_BG if ri % 2 else WHT_BG)
                txt_color = RED_TXT if flagged else "000000"
                peak_fi   = int(np.argmax(np.abs(Xz[ri]))) if ri < Xz.shape[0] else 0

                vals = [
                    row.get("track_id", "?"),
                    round(row.get("z_score_max", 0), 3),
                    "ANOMALY" if flagged else "Normal",
                    row.get("top_feature", "?"),
                    round(float(Xz[ri, peak_fi]), 3) if ri < Xz.shape[0] else 0,
                    n_live,
                    "YES" if flagged else "no",
                ]
                for ci, val in enumerate(vals, 1):
                    cell = ws1.cell(row=r_xl, column=ci, value=val)
                    cell.fill      = _hdr_fill(bg)
                    cell.font      = _font(bold=flagged, color=txt_color)
                    cell.alignment = _align(h="left" if ci in (1, 4) else "center",
                                            wrap=True)
                    cell.border    = _border()
                ws1.row_dimensions[r_xl].height = 18

            # Freeze pane below headers
            ws1.freeze_panes = "A6"

            # ══════════════════════════════════════════════════════════════
            # SHEET 2 — Z-SCORE HEATMAP  (features with signal only)
            # ══════════════════════════════════════════════════════════════
            ws2 = wb.create_sheet("Z-Score Heatmap")
            ws2.sheet_view.showGridLines = False

            # Mirror the GUI heatmap: top-50 highest-scoring features by max |z|.
            # Use the cached selection when available so chart and GUI are identical.
            _top50_idx   = getattr(self, "_ai_top50_feat_idx",   None)
            _top50_names = getattr(self, "_ai_top50_feat_names", None)
            if _top50_idx is not None and _top50_names is not None:
                sig_idx   = np.array(_top50_idx, dtype=int)
                sig_feats = _top50_names
            else:
                # Fallback: compute top-50 from the result z-matrix
                peak_z   = np.abs(Xz).max(axis=0)
                _TOP_N   = 30
                n_avail  = Xz.shape[1]
                if n_avail <= _TOP_N:
                    sig_idx = np.arange(n_avail)
                else:
                    sig_idx = np.sort(np.argsort(peak_z)[-_TOP_N:])
                sig_feats = [feat_cols[i] for i in sig_idx]
            Xz_sig    = Xz[:, sig_idx]

            # Banner
            ws2.merge_cells(f"A1:{get_column_letter(n_veh + 1)}1")
            c = ws2["A1"]
            c.value = f"Z-Score Heatmap  |  Top {len(sig_feats)} highest-scoring features  (threshold {threshold:.2f}σ)"
            c.font  = _font(bold=True, color="FFFFFF", size=12)
            c.fill  = _hdr_fill(NAVY)
            c.alignment = _align()
            ws2.row_dimensions[1].height = 24

            # Header row: vehicle IDs
            ws2.cell(row=2, column=1, value="Feature").font = _font(bold=True, color="FFFFFF")
            ws2.cell(row=2, column=1).fill = _hdr_fill("2C4F7C")
            ws2.cell(row=2, column=1).alignment = _align()
            ws2.column_dimensions["A"].width = 38
            for vi, row in enumerate(rows):
                cell = ws2.cell(row=2, column=vi + 2, value=row.get("track_id", f"V{vi}"))
                cell.font      = _font(bold=True, color="FFFFFF")
                cell.fill      = _hdr_fill("8B0000" if row.get("flagged") else "2C4F7C")
                cell.alignment = _align()
                ws2.column_dimensions[get_column_letter(vi + 2)].width = 13

            # Feature rows
            for fi, (feat, fi_orig) in enumerate(zip(sig_feats, sig_idx)):
                r_xl = fi + 3
                # Feature name cell
                cell = ws2.cell(row=r_xl, column=1, value=feat)
                cell.font      = _font(size=9)
                cell.fill      = _hdr_fill(ALT_BG if fi % 2 else WHT_BG)
                cell.alignment = _align(h="left")
                cell.border    = _border()
                ws2.row_dimensions[r_xl].height = 15

                for vi in range(n_veh):
                    z = float(Xz_sig[vi, fi])
                    cell = ws2.cell(row=r_xl, column=vi + 2,
                                    value=round(z, 2))
                    cell.number_format = "0.00"
                    cell.alignment     = _align()
                    cell.border        = _border()
                    cell.font          = _font(
                        bold=(abs(z) > threshold),
                        color=RED_TXT if z > threshold else
                              ("1F3864" if z < -threshold else "000000"),
                        size=9
                    )
                    # Manual cell shading based on z-score intensity
                    intensity = min(abs(z) / 8.0, 1.0)
                    if z > 0.3:
                        r_val = 255
                        g_val = int(255 * (1 - intensity * 0.85))
                        b_val = int(255 * (1 - intensity * 0.85))
                    elif z < -0.3:
                        r_val = int(255 * (1 - intensity * 0.85))
                        g_val = int(255 * (1 - intensity * 0.85))
                        b_val = 255
                    else:
                        r_val = g_val = b_val = 255
                    hex_fill = f"FF{r_val:02X}{g_val:02X}{b_val:02X}"
                    cell.fill = PatternFill("solid", fgColor=hex_fill)

            ws2.freeze_panes = "B3"

            # ══════════════════════════════════════════════════════════════
            # SHEET 3 — FULL Z-SCORE TABLE  (all features)
            # ══════════════════════════════════════════════════════════════
            ws3 = wb.create_sheet("All Feature Z-Scores")
            ws3.sheet_view.showGridLines = False

            ws3.merge_cells(f"A1:{get_column_letter(n_veh + 1)}1")
            c = ws3["A1"]
            c.value = f"Full Z-Score Matrix  |  All {n_feat} model features  |  {scenario}"
            c.font  = _font(bold=True, color="FFFFFF", size=12)
            c.fill  = _hdr_fill(NAVY)
            c.alignment = _align()
            ws3.row_dimensions[1].height = 24

            ws3.cell(row=2, column=1, value="Feature").font = _font(bold=True, color="FFFFFF")
            ws3.cell(row=2, column=1).fill = _hdr_fill("2C4F7C")
            ws3.cell(row=2, column=1).alignment = _align()
            ws3.column_dimensions["A"].width = 38
            for vi, row in enumerate(rows):
                cell = ws3.cell(row=2, column=vi + 2, value=row.get("track_id", f"V{vi}"))
                cell.font      = _font(bold=True, color="FFFFFF")
                cell.fill      = _hdr_fill("8B0000" if row.get("flagged") else "2C4F7C")
                cell.alignment = _align()
                ws3.column_dimensions[get_column_letter(vi + 2)].width = 13

            for fi, feat in enumerate(feat_cols):
                r_xl = fi + 3
                cell = ws3.cell(row=r_xl, column=1, value=feat)
                cell.font      = _font(size=8)
                cell.fill      = _hdr_fill(ALT_BG if fi % 2 else WHT_BG)
                cell.alignment = _align(h="left")
                cell.border    = _border()
                ws3.row_dimensions[r_xl].height = 13

                for vi in range(n_veh):
                    z    = float(Xz[vi, fi]) if fi < Xz.shape[1] else 0.0
                    cell = ws3.cell(row=r_xl, column=vi + 2, value=round(z, 3))
                    cell.number_format = "0.000"
                    cell.alignment     = _align()
                    cell.border        = _border()
                    cell.font          = _font(
                        bold=(abs(z) > threshold),
                        color=RED_TXT if abs(z) > threshold else "000000",
                        size=8
                    )
                    if abs(z) > threshold:
                        cell.fill = _hdr_fill(RED_BG)
                    elif abs(z) > threshold * 0.5:
                        cell.fill = _hdr_fill("FFE8CC")
                    else:
                        cell.fill = _hdr_fill(ALT_BG if fi % 2 else WHT_BG)

            ws3.freeze_panes = "B3"

            # ══════════════════════════════════════════════════════════════
            # SHEET 4 — FEATURE SCORES  (vehicles × features, z-score vectors)
            #   One row per vehicle; one column per model feature (all n_feat).
            #   This gives a "162-long vector for each car" as requested.
            # ══════════════════════════════════════════════════════════════
            ws4 = wb.create_sheet("Feature Scores")
            ws4.sheet_view.showGridLines = False

            # Banner
            ws4.merge_cells(f"A1:{get_column_letter(n_feat + 1)}1")
            c = ws4["A1"]
            c.value = (
                f"Feature Score Vectors  |  {n_veh} vehicles × {n_feat} features  |  "
                f"Z-scores (σ from training mean)  |  {scenario}"
            )
            c.font  = _font(bold=True, color="FFFFFF", size=12)
            c.fill  = _hdr_fill(NAVY)
            c.alignment = _align()
            ws4.row_dimensions[1].height = 24

            # Header row: "Vehicle" label + all feature names
            ws4.cell(row=2, column=1, value="Vehicle").font = _font(bold=True, color="FFFFFF")
            ws4.cell(row=2, column=1).fill      = _hdr_fill("2C4F7C")
            ws4.cell(row=2, column=1).alignment = _align()
            ws4.column_dimensions["A"].width    = 14
            ws4.row_dimensions[2].height        = 40   # tall row for rotated headers
            for fi, feat in enumerate(feat_cols):
                cell = ws4.cell(row=2, column=fi + 2, value=feat)
                cell.font      = _font(bold=True, color="FFFFFF", size=8)
                cell.fill      = _hdr_fill("2C4F7C")
                cell.alignment = Alignment(
                    horizontal="center", vertical="bottom",
                    text_rotation=75, wrap_text=False
                )
                ws4.column_dimensions[get_column_letter(fi + 2)].width = 7

            # One data row per vehicle
            for vi, row in enumerate(rows):
                r_xl    = vi + 3
                flagged = row.get("flagged", False)
                # Vehicle ID cell
                cell = ws4.cell(row=r_xl, column=1,
                                value=row.get("track_id", f"V{vi}"))
                cell.font      = _font(bold=flagged,
                                       color=RED_TXT if flagged else "000000", size=9)
                cell.fill      = _hdr_fill(RED_BG if flagged else
                                           (ALT_BG if vi % 2 else WHT_BG))
                cell.alignment = _align()
                cell.border    = _border()
                ws4.row_dimensions[r_xl].height = 15

                # Z-score for each feature
                for fi in range(n_feat):
                    z    = float(Xz[vi, fi]) if fi < Xz.shape[1] else 0.0
                    cell = ws4.cell(row=r_xl, column=fi + 2, value=round(z, 3))
                    cell.number_format = "0.000"
                    cell.alignment     = _align()
                    cell.border        = _border()
                    cell.font          = _font(
                        bold=(abs(z) > threshold),
                        color=RED_TXT if abs(z) > threshold else "000000",
                        size=8
                    )
                    intensity = min(abs(z) / 8.0, 1.0)
                    if z > 0.3:
                        r_val = 255
                        g_val = int(255 * (1 - intensity * 0.85))
                        b_val = int(255 * (1 - intensity * 0.85))
                    elif z < -0.3:
                        r_val = int(255 * (1 - intensity * 0.85))
                        g_val = int(255 * (1 - intensity * 0.85))
                        b_val = 255
                    else:
                        r_val = g_val = b_val = 255
                    hex_fill = f"FF{r_val:02X}{g_val:02X}{b_val:02X}"
                    cell.fill = PatternFill("solid", fgColor=hex_fill)

            ws4.freeze_panes = "B3"

            # ══════════════════════════════════════════════════════════════
            # SHEET 5 — CHARTS
            # ══════════════════════════════════════════════════════════════
            ws4 = wb.create_sheet("Charts")
            ws4.sheet_view.showGridLines = False

            ws4.merge_cells("A1:P1")
            c = ws4["A1"]
            c.value = f"Anomaly Intel Charts  |  {scenario}"
            c.font  = _font(bold=True, color="FFFFFF", size=13)
            c.fill  = _hdr_fill(NAVY)
            c.alignment = _align()
            ws4.row_dimensions[1].height = 26

            chart_figs = [
                (self._ai_fig_heat,  "Z-Score Heatmap"),
                (self._ai_fig_bars,  "Score Ranking"),
                (self._ai_fig_dist,  "Feature Distributions"),
                (self._ai_fig_cusum, "CUSUM Temporal Analysis"),
            ]
            row_offset = 3
            # Export each figure at its actual GUI size, but at 200 dpi so the
            # PNG is sharp and matches exactly what is shown on screen.
            EXPORT_DPI   = 200
            IMG_PX_W     = 1100   # width the image will be displayed in Excel
            IMG_PX_H     = 700    # height the image will be displayed in Excel
            ROWS_PER_IMAGE = int(IMG_PX_H / 15) + 4

            # Set chart sheet column widths so the image has room
            for col_letter in "ABCDEFGHIJKLMNO":
                ws4.column_dimensions[col_letter].width = 18

            for fig_obj, title in chart_figs:
                if fig_obj is None:
                    continue
                # Label row
                lbl_cell = ws4.cell(row=row_offset, column=1, value=title)
                lbl_cell.font      = _font(bold=True, color="2C4F7C", size=12)
                lbl_cell.alignment = _align(h="left")
                ws4.row_dimensions[row_offset].height = 22
                row_offset += 1

                # Save at current figure size — no resize so layout stays identical
                buf = io.BytesIO()
                fig_obj.savefig(buf, format="png", dpi=EXPORT_DPI,
                                bbox_inches="tight", facecolor="white",
                                pad_inches=0.15)
                buf.seek(0)

                img = XLImage(buf)
                img.width  = IMG_PX_W
                img.height = IMG_PX_H
                ws4.add_image(img, f"A{row_offset}")
                # Reserve enough rows for image + 2-row gap
                for r in range(row_offset, row_offset + ROWS_PER_IMAGE):
                    ws4.row_dimensions[r].height = 15
                row_offset += ROWS_PER_IMAGE

            # ══════════════════════════════════════════════════════════════
            # Feature Catalog sheet
            # ══════════════════════════════════════════════════════════════
            try:
                from anomaly_model.make_features_excel import (
                    _build_catalog_sheet, FEATURE_CATALOG_RICH)
                ws_cat = wb.create_sheet("Feature Catalog")
                _build_catalog_sheet(ws_cat, feat_cols)
            except Exception:
                pass  # catalog is best-effort; don't block the save

            # ══════════════════════════════════════════════════════════════
            # Save workbook
            # ══════════════════════════════════════════════════════════════
            wb.save(xl_path)
            messagebox.showinfo(
                "Export Complete",
                f"Saved:\n{os.path.basename(xl_path)}\n\nin:\n{os.path.dirname(xl_path)}\n\n"
                f"Sheets: Summary · Z-Score Heatmap (top 50) · All Feature Z-Scores · Feature Scores · Charts · Feature Catalog"
            )

        except Exception as exc:
            import traceback
            messagebox.showerror(
                "Export Failed", f"{exc}\n\n{traceback.format_exc()[-800:]}"
            )

    def _derive_sensor_acc_rows(self, topic: str, sensor_lookup: dict = None):
        _TOPIC_SOURCE = {"sensor.gps": "gps", "sensor.camera": "cam", "sensor.das": "das"}
        source_kind = _TOPIC_SOURCE.get(topic, "")
        events = [ev for ev in self._all_events if getattr(ev, "topic", "") == topic]
        events.sort(key=lambda ev: (str(ev.payload.get("sensor_id", "")), str(ev.payload.get("vehicle_id", "")), float(ev.payload.get("t", 0.0) or 0.0)))
        grouped = {}
        for ev in events:
            p = ev.payload or {}
            key = (str(p.get("sensor_id", "")), str(p.get("vehicle_id", "")))
            grouped.setdefault(key, []).append(p)
        rows = []
        for (sid, vid), seq in grouped.items():
            acc_vals = []
            for idx, p in enumerate(seq):
                a = 0.0
                if idx >= 1:
                    dt = float(p.get("t", 0.0) or 0.0) - float(seq[idx-1].get("t", 0.0) or 0.0)
                    if dt > 1e-6:
                        a = (float(p.get("speed_mps", 0.0) or 0.0) - float(seq[idx-1].get("speed_mps", 0.0) or 0.0)) / dt
                acc_vals.append(a)
            # light smoothing to avoid noisy one-step derivative spikes
            for idx, p in enumerate(seq):
                if len(acc_vals) == 1:
                    a_s = acc_vals[0]
                elif idx == 0:
                    a_s = 0.5 * (acc_vals[0] + acc_vals[1])
                elif idx == len(acc_vals)-1:
                    a_s = 0.5 * (acc_vals[-2] + acc_vals[-1])
                else:
                    a_s = (acc_vals[idx-1] + acc_vals[idx] + acc_vals[idx+1]) / 3.0
                x_val = float(p.get('x', 0.0) or 0.0)
                y_val = float(p.get('y', 0.0) or 0.0)
                gid = sensor_lookup.get((source_kind, x_val, y_val), "") if sensor_lookup else ""
                rows.append((
                    f"{float(p.get('t', 0.0) or 0.0):.2f}",
                    sid,
                    gid,
                    vid,
                    f"{x_val:.2f}",
                    f"{y_val:.2f}",
                    f"{float(p.get('speed_mps', 0.0) or 0.0):.2f}",
                    f"{a_s:.3f}",
                    f"{float(p.get('sigma_m', 0.0) or 0.0):.3f}",
                    f"{float(p.get('confidence', 0.0) or 0.0):.2f}",
                ))
        rows.sort(key=lambda r: (r[1], r[3], float(r[0])))
        return rows

    def _derive_das_rows(self, sensor_lookup: dict = None):
        """Build DAS display rows.

        Speed is estimated via a causal rolling OLS fit over the last
        _DAS_OLS_N (time, fiber_position_m) samples for each
        (sensor_id, vehicle_id) track.  Acceleration is not shown.
        """
        import numpy as _np
        _DAS_OLS_N = 10  # causal window — last N samples

        events = [ev for ev in self._all_events if getattr(ev, "topic", "") == "sensor.das"]
        events.sort(key=lambda ev: (
            str(ev.payload.get("sensor_id", "")),
            str(ev.payload.get("vehicle_id", "")),
            float(ev.payload.get("t", 0.0) or 0.0),
        ))
        grouped = {}
        for ev in events:
            p = ev.payload or {}
            key = (str(p.get("sensor_id", "")), str(p.get("vehicle_id", "")))
            grouped.setdefault(key, []).append(p)

        rows = []
        for (sid, vid), seq in grouped.items():
            for idx, p in enumerate(seq):
                # Causal window: up to _DAS_OLS_N points ending at current sample
                win = seq[max(0, idx - _DAS_OLS_N + 1): idx + 1]
                speed = 0.0
                if len(win) >= 2:
                    t_arr = [float(w.get("t", 0.0) or 0.0) for w in win]
                    x_arr = [float(w.get("x", 0.0) or 0.0) for w in win]
                    y_arr = [float(w.get("y", 0.0) or 0.0) for w in win]
                    try:
                        vx_fit = float(_np.polyfit(t_arr, x_arr, 1)[0])
                        vy_fit = float(_np.polyfit(t_arr, y_arr, 1)[0])
                        speed = float(_np.hypot(vx_fit, vy_fit))
                    except Exception:
                        speed = 0.0
                x_val = float(p.get('x', 0.0) or 0.0)
                y_val = float(p.get('y', 0.0) or 0.0)
                gid = sensor_lookup.get(("das", x_val, y_val), "") if sensor_lookup else ""
                row = (
                    f"{float(p.get('t', 0.0) or 0.0):.2f}",
                    sid,
                    gid,
                    vid,
                    f"{x_val:.2f}",
                    f"{y_val:.2f}",
                    f"{float(p.get('fiber_position_m', 0.0) or 0.0):.2f}",
                    f"{speed:.2f}",
                    f"{float(p.get('fiber_angle_rad', 0.0) or 0.0):.3f}",
                    f"{float(p.get('snr', 0.0) or 0.0):.2f}",
                )
                rows.append(row)
        rows.sort(key=lambda r: (r[1], r[3], float(r[0])))
        return rows

    def _plot_das_trajectories(self):
        """Draw (t, fiber_position_m) scatter per vehicle for each DAS fiber.

        One subplot per fiber; each vehicle gets a distinct color.
        Called after _derive_das_rows at stop time.
        """
        if not _MPL_OK or getattr(self, "_das_fig", None) is None:
            return

        fig = self._das_fig
        fig.clf()

        events = [ev for ev in self._all_events if getattr(ev, "topic", "") == "sensor.das"]
        if not events:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No DAS data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self._das_canvas.draw()
            return

        # Group: fiber → vehicle → list of (t, fiber_position_m)
        by_fiber: dict = {}
        for ev in events:
            p = ev.payload or {}
            sid = str(p.get("sensor_id", ""))
            vid = str(p.get("vehicle_id", ""))
            t_val = float(p.get("t", 0.0) or 0.0)
            s_val = float(p.get("fiber_position_m", 0.0) or 0.0)
            by_fiber.setdefault(sid, {}).setdefault(vid, []).append((t_val, s_val))

        fibers = sorted(by_fiber.keys())
        n_fibers = len(fibers)

        # Give every fiber subplot its own 2.2-inch tall panel so labels,
        # tick marks and legends never overlap.  The outer scroll container
        # (created in make_das_table) makes the extra height accessible.
        HEIGHT_PER_FIBER = 2.2   # inches per subplot — increase if still cramped
        fig_h = max(2.5, n_fibers * HEIGHT_PER_FIBER)

        # Get the current pixel width of the scroll canvas so the figure
        # fills the available width; fall back to 8 inches if not mapped yet.
        try:
            sc = getattr(self, "_das_scroll_canvas", None)
            canvas_px_w = sc.winfo_width() if sc and sc.winfo_width() > 10 else 720
            fig_w = max(6.0, canvas_px_w / fig.get_dpi())
        except Exception:
            fig_w = 8.0

        fig.set_size_inches(fig_w, fig_h)

        # Resize the Tk canvas widget to match the figure's new pixel dimensions.
        # set_size_inches() changes the figure's internal size but does NOT
        # automatically resize the Tk widget — without this the widget stays at
        # its original small height and all subplots are crammed into it.
        try:
            dpi = fig.get_dpi()
            new_w_px = max(1, int(fig_w * dpi))
            new_h_px = max(1, int(fig_h * dpi))
            self._das_canvas.get_tk_widget().config(
                width=new_w_px, height=new_h_px
            )
        except Exception:
            pass

        # Distinct colors — cycle if more than 10 vehicles
        _COLORS = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        for i, sid in enumerate(fibers):
            ax = fig.add_subplot(n_fibers, 1, i + 1)
            veh_tracks = by_fiber[sid]
            for j, (vid, pts) in enumerate(sorted(veh_tracks.items())):
                pts_sorted = sorted(pts, key=lambda pt: pt[0])
                t_vals = [pt[0] for pt in pts_sorted]
                s_vals = [pt[1] for pt in pts_sorted]
                color = _COLORS[j % len(_COLORS)]
                ax.plot(t_vals, s_vals, ".", markersize=3, color=color, label=vid)
            ax.set_title(f"Fiber: {sid}", fontsize=9, pad=2)
            ax.set_xlabel("t (s)", fontsize=8)
            ax.set_ylabel("fiber pos (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            if veh_tracks:
                ax.legend(fontsize=7, loc="upper left", markerscale=2)

        try:
            fig.tight_layout(pad=0.8, h_pad=1.5)
        except Exception:
            pass

        self._das_canvas.draw()

        # Tell the scroll container about the new figure height so the
        # scrollbar range is immediately correct.
        try:
            if getattr(self, "_das_scroll_canvas", None):
                self._das_scroll_canvas.configure(
                    scrollregion=self._das_scroll_canvas.bbox("all")
                )
                # Scroll back to the top when the plot is refreshed
                self._das_scroll_canvas.yview_moveto(0.0)
        except Exception:
            pass

    def _build_rmse_rows(self):
        import math

        def sensor_rmse(topic: str, label: str):
            xs, ys, ps = [], [], []
            n = 0
            for ev in self._all_events:
                if getattr(ev, "topic", "") != topic:
                    continue
                p = ev.payload or {}
                if p.get("x_true") is None or p.get("y_true") is None:
                    continue
                dx = float(p.get("x", 0.0) or 0.0) - float(p.get("x_true", 0.0) or 0.0)
                dy = float(p.get("y", 0.0) or 0.0) - float(p.get("y_true", 0.0) or 0.0)
                xs.append(dx*dx); ys.append(dy*dy); ps.append(dx*dx+dy*dy); n += 1
            if not n:
                return (label, "0", "", "", "", "no truth-aligned samples")
            return (label, str(n), f"{math.sqrt(sum(xs)/n):.2f}", f"{math.sqrt(sum(ys)/n):.2f}", f"{math.sqrt(sum(ps)/n):.2f}", "position RMSE over available detections")

        rows = [
            sensor_rmse("sensor.gps", "GPS"),
            sensor_rmse("sensor.camera", "Camera"),
            sensor_rmse("sensor.das", "DAS"),
        ]
        kx, ky, kp, kn = build_kalman_rmse(self._all_events)
        if kn:
            rows.append(("Kalman", str(kn), f"{kx:.2f}", f"{ky:.2f}", f"{kp:.2f}", "fusion estimate vs nearest ground truth"))
        else:
            rows.append(("Kalman", "0", "", "", "", "no truth-aligned samples"))
        return rows

    def _export_current_table(self):
        if not self._data_ready:
            messagebox.showinfo("Export", "No dataset yet. Press Start then Stop first.")
            return

        tab = self.data_nb.index("current")
        mapping = {
            0: ("summary.csv", self.tv_summary),
            1: ("vehicles.csv", self.tv_veh),
            2: ("gps.csv", self.tv_gps),
            3: ("cameras.csv", self.tv_cam),
            4: ("das.csv", self.tv_das),
            5: ("kalman.csv", self.tv_kalman),
            6: ("rmse.csv", self.tv_rmse),
            7: ("issues.csv", self.tv_anom),
        }
        fname, tv = mapping.get(tab, ("table.csv", self.tv_summary))

        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        out = Path(folder) / fname
        self._export_treeview_csv(tv, out)
        self.status.set(f"Exported: {out}")

    def _export_all_tables(self):
        if not self._data_ready:
            messagebox.showinfo("Export", "No dataset yet. Press Start then Stop first.")
            return

        # Guard: only one export at a time.
        if getattr(self, "_export_in_progress", False):
            messagebox.showinfo("Export", "An export is already in progress. Please wait.")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")

        if _OPENPYXL_OK:
            # Ask for the destination file while still on the main thread
            # (file dialogs must be called from the main thread).
            out_path = filedialog.asksaveasfilename(
                title="Save Excel export",
                defaultextension=".xlsx",
                initialfile=f"{getattr(self, '_scene_name', 'simstudio')}_export_{ts}.xlsx",
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            )
            if not out_path:
                return
            out_path = Path(out_path)

            # ── Snapshot all treeview data on the main thread ──────────────
            # tv.get_children() / tv.item() are Tkinter calls and MUST happen
            # here.  The resulting _FakeTv objects are plain Python and can be
            # safely read from the background thread.
            # Phase 6: tracker treeviews are snapshotted on the main thread
            # and passed through to ``export_workbook``.  When absent
            # (e.g. legacy tests that don't build the Tracks/Diagnostics
            # tabs) the kwargs default to None and the sheets are skipped.
            snaps = dict(
                tv_summary=_snapshot_tv(self.tv_summary),
                tv_veh    =_snapshot_tv(self.tv_veh),
                tv_gps    =_snapshot_tv(self.tv_gps),
                tv_cam    =_snapshot_tv(self.tv_cam),
                tv_das    =_snapshot_tv(self.tv_das),
                tv_kalman =_snapshot_tv(self.tv_kalman),
                tv_rmse   =_snapshot_tv(self.tv_rmse),
                tv_anom   =_snapshot_tv(self.tv_anom),
                tv_tracks       =_snapshot_tv(self.tv_tracks)       if hasattr(self, "tv_tracks")       else None,
                tv_diag         =_snapshot_tv(self.tv_diag)         if hasattr(self, "tv_diag")         else None,
                tv_diag_summary =_snapshot_tv(self.tv_diag_summary) if hasattr(self, "tv_diag_summary") else None,
            )

            # Snapshot raw events + world for the audit sidecar files written
            # next to the xlsx.  The thread reads these copies, never touching
            # tkinter state directly.
            events_for_audit = list(self._all_events)
            world_for_audit = getattr(self, "world", None)

            self._export_in_progress = True
            self.status.set("Exporting… (working in background)")

            def _do_xlsx_export():
                try:
                    if _EXPORT_OK:
                        _export_workbook(out_path, **snaps)
                    else:
                        # Fallback to plain openpyxl if the styled module is unavailable.
                        wb = openpyxl.Workbook()
                        wb.remove(wb.active)
                        for sheet_name, tv in [
                            ("Summary",  snaps["tv_summary"]),
                            ("Vehicles", snaps["tv_veh"]),
                            ("GPS",      snaps["tv_gps"]),
                            ("Cameras",  snaps["tv_cam"]),
                            ("DAS",      snaps["tv_das"]),
                            ("Kalman",   snaps["tv_kalman"]),
                            ("RMSE",     snaps["tv_rmse"]),
                            ("Issues",   snaps["tv_anom"]),
                        ]:
                            ws = wb.create_sheet(title=sheet_name)
                            self._treeview_to_sheet(tv, ws)
                        wb.save(out_path)
                    # Sidecar: write the tracking-audit CSV/PNG bundle next
                    # to the xlsx so "Export All" delivers everything in one
                    # action.  Failures here never block the xlsx — just log.
                    _sidecar_ok = False
                    try:
                        sidecar = out_path.parent / f"{out_path.stem}_tracking_audit"
                        # progress_cb runs on the background thread; schedule
                        # each update onto the Tk main thread via self.after().
                        def _audit_progress(msg: str) -> None:
                            self.after(0, lambda m=msg: self.status.set(f"Exporting… {m}"))
                        _audit_mod.export_all(
                            events_for_audit, sidecar,
                            world=world_for_audit,
                            progress_cb=_audit_progress,
                        )
                        msg = f"Exported: {out_path}  (+ tracking_audit/ folder)"
                        _sidecar_ok = True
                    except Exception as _exc:
                        msg = f"Exported: {out_path}  (tracking-audit sidecar failed: {_exc})"
                        sidecar = None
                    self.after(0, lambda m=msg: self.status.set(m))
                    # Track the last successful audit dir and fire any pending
                    # post-export callback (e.g. "Save & Analyse" flow).
                    if _sidecar_ok and sidecar is not None:
                        self.after(0, lambda d=sidecar: setattr(self, "_last_audit_dir", d))
                        self.after(0, lambda p=out_path: setattr(self, "_last_xlsx_export_path", p))
                        _cb = getattr(self, "_post_export_cb", None)
                        if _cb is not None:
                            self._post_export_cb = None
                            self.after(200, lambda d=sidecar: _cb(d))
                except Exception as exc:
                    self.after(0, lambda: self.status.set(f"Export failed: {exc}"))
                finally:
                    self._export_in_progress = False

            threading.Thread(target=_do_xlsx_export, daemon=True).start()

        else:
            # Fallback: CSV files in a folder (openpyxl not installed).
            folder = filedialog.askdirectory(title="Choose export root folder")
            if not folder:
                return
            out_dir = Path(folder) / f"simstudio_export_{ts}"

            # ── Snapshot treeviews and events on the main thread ───────────
            snaps = dict(
                summary=_snapshot_tv(self.tv_summary),
                veh    =_snapshot_tv(self.tv_veh),
                gps    =_snapshot_tv(self.tv_gps),
                cam    =_snapshot_tv(self.tv_cam),
                das    =_snapshot_tv(self.tv_das),
                kalman =_snapshot_tv(self.tv_kalman),
                rmse   =_snapshot_tv(self.tv_rmse),
                anom   =_snapshot_tv(self.tv_anom),
            )
            # Shallow-copy the events list so the background thread holds a
            # stable reference even if the main thread later clears it.
            events_copy = list(self._all_events)

            self._export_in_progress = True
            self.status.set("Exporting… (working in background)")

            def _do_csv_export():
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    self._export_treeview_csv(snaps["summary"], out_dir / "summary.csv")
                    self._export_treeview_csv(snaps["veh"],     out_dir / "vehicles.csv")
                    self._export_treeview_csv(snaps["gps"],     out_dir / "gps.csv")
                    self._export_treeview_csv(snaps["cam"],     out_dir / "cameras.csv")
                    self._export_treeview_csv(snaps["das"],     out_dir / "das.csv")
                    self._export_treeview_csv(snaps["kalman"],  out_dir / "kalman.csv")
                    self._export_treeview_csv(snaps["rmse"],    out_dir / "rmse.csv")
                    self._export_treeview_csv(snaps["anom"],    out_dir / "issues.csv")
                    raw = out_dir / "events.jsonl"
                    with raw.open("w", encoding="utf-8") as fp:
                        for ev in events_copy:
                            fp.write(
                                json.dumps({"topic": ev.topic, "ts": ev.ts, "payload": ev.payload}) + "\n"
                            )
                    self.after(0, lambda: self.status.set(f"Exported all: {out_dir}"))
                except Exception as exc:
                    self.after(0, lambda: self.status.set(f"Export failed: {exc}"))
                finally:
                    self._export_in_progress = False

            threading.Thread(target=_do_csv_export, daemon=True).start()

    # ----- Tracking audit / trajectory exporter ----------------------------
    #
    # Surgical addition: re-uses the existing self._all_events stream and
    # the running self.world reference.  All work runs on a background
    # thread so the GUI stays responsive while matplotlib renders the PNG.
    # Zero changes to the tracker / Kalman / sensor pipeline.

    def _export_tracking_audit(self):
        """Write audit, per-track trajectory CSV/PNG, and coverage summary."""
        if not getattr(self, "_data_ready", False) or not self._all_events:
            messagebox.showinfo(
                "Tracking audit",
                "No dataset yet. Press Start, run for a moment, then Stop.",
            )
            return
        if getattr(self, "_export_in_progress", False):
            messagebox.showinfo(
                "Tracking audit",
                "An export is already in progress. Please wait.",
            )
            return

        folder = filedialog.askdirectory(title="Choose tracking-audit output folder")
        if not folder:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(folder) / f"tracking_audit_{ts}"

        events_copy = list(self._all_events)
        world_ref = getattr(self, "world", None)

        self._export_in_progress = True
        self.status.set("Tracking audit: working in background…")

        def _run():
            try:
                summary = _audit_mod.export_all(
                    events_copy, out_dir, world=world_ref,
                )
                msg = (
                    f"Tracking audit written to {out_dir}\n"
                    f"  audit rows : {summary['n_audit_rows']}\n"
                    f"  traj rows  : {summary['n_traj_rows']}\n"
                    f"  tracks     : {summary['n_gids']}"
                )
                self.after(0, lambda: self.status.set(msg.replace("\n", " | ")))
            except Exception as exc:
                self.after(
                    0,
                    lambda exc=exc: self.status.set(f"Tracking audit failed: {exc}"),
                )
            finally:
                self._export_in_progress = False

        threading.Thread(target=_run, daemon=True).start()

    def _treeview_to_sheet(self, tv, ws):
        """Write a Treeview's contents into an openpyxl worksheet with a styled header row."""
        cols = tv["columns"]

        # Header row
        header_font = Font(name="Arial", bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", start_color="2E4057")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for col_idx, col_name in enumerate(cols, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align

        # Data rows
        data_font = Font(name="Arial", size=10)
        for row_idx, iid in enumerate(tv.get_children(), start=2):
            for col_idx, value in enumerate(tv.item(iid, "values"), start=1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.font = data_font

        # Auto-fit column widths (capped at 40)
        for col_idx, col_name in enumerate(cols, start=1):
            col_letter = openpyxl.utils.get_column_letter(col_idx)
            max_len = len(str(col_name))
            for iid in tv.get_children():
                val = str(tv.item(iid, "values")[col_idx - 1]) if col_idx - 1 < len(tv.item(iid, "values")) else ""
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(40, max_len + 3)

        # Freeze the header row
        ws.freeze_panes = "A2"

    def _export_treeview_csv(self, tv, path: Path):
        cols = tv["columns"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for iid in tv.get_children():
                w.writerow(tv.item(iid, "values"))

# ---------------- Map Import (OSM + Overpass) ----------------

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def _latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Local equirectangular projection around (lat0, lon0). Returns meters."""
    R = 6378137.0
    x = _deg2rad(lon - lon0) * R * math.cos(_deg2rad(lat0))
    y = _deg2rad(lat - lat0) * R
    return (x, -y)  # invert Y for screen-friendly (north up)


class _ThreeOptionDialog(tk.Toplevel):
    def __init__(self, parent, title: str, message: str, opt1: str, opt2: str, opt3: str):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.choice = None
        self.transient(parent)
        self.grab_set()
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')
        ttk.Label(frm, text=message, wraplength=420, justify='left').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))
        def set_choice(v):
            self.choice = v
            self.destroy()
        ttk.Button(frm, text=opt1, command=lambda: set_choice('opt1')).grid(row=1, column=0, sticky='ew', padx=4)
        ttk.Button(frm, text=opt2, command=lambda: set_choice('opt2')).grid(row=1, column=1, sticky='ew', padx=4)
        ttk.Button(frm, text=opt3, command=lambda: set_choice('opt3')).grid(row=1, column=2, sticky='ew', padx=4)
        for c in range(3):
            frm.columnconfigure(c, weight=1)
        self.update_idletasks()
        # center
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f'+{x}+{y}')


class MapImportWindow(tk.Toplevel):
    """Lightweight OSM tile viewer with bbox selection.

    Notes:
    - Requires internet access.
    - Uses public OSM tile server and Nominatim; keep reasonable use.
    """

    def __init__(self, parent: 'SimStudioApp'):
        super().__init__(parent)
        self.app = parent
        self.title('Import Roads From Map')
        self.geometry('980x700')
        self.minsize(860, 620)
        self.protocol('WM_DELETE_WINDOW', self._close)

        # Map state
        self.zoom = 16
        self.center_lat = 32.0853
        self.center_lon = 34.7818
        self._tile_cache = {}
        # PhotoImage cache: keyed by (zoom, x, y).
        # Each Pillow Image is converted to an ImageTk.PhotoImage exactly once
        # and reused on all subsequent redraws.  Entries are invalidated when a
        # tile is updated from the background loader.
        self._tk_tile_cache: dict = {}
        self._tile_inflight = set()
        self._tile_req_q: Queue = Queue()
        self._tile_res_q: Queue = Queue()
        self._tile_stop = threading.Event()
        self._tile_redraw_scheduled = False
        self._tile_loader = threading.Thread(target=self._tile_loader_loop, daemon=True)
        self._tile_loader.start()

        # Poll for loaded tiles without blocking the UI.
        self.after(30, self._process_loaded_tiles)
        self._img_refs = []

        # Selection
        self._select_mode = False
        self._sel_start = None
        self._sel_end = None

        # UI
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(side='top', fill='x')
        ttk.Label(top, text='Search').pack(side='left')
        self.q = tk.StringVar(value='')
        ent = ttk.Entry(top, textvariable=self.q, width=42)
        ent.pack(side='left', padx=(6, 8))
        ttk.Button(top, text='Go', command=self._search).pack(side='left')
        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Button(top, text='Select Area', command=self._toggle_select).pack(side='left')

        # Layer toggles (import)
        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=10)
        self._import_roads_var = tk.BooleanVar(value=True)
        self._import_cameras_var = tk.BooleanVar(value=False)
        self._import_gps_var = tk.BooleanVar(value=False)
        self._import_das_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='Roads', variable=self._import_roads_var).pack(side='left')
        ttk.Checkbutton(top, text='Cameras', variable=self._import_cameras_var).pack(side='left', padx=(6, 0))
        ttk.Checkbutton(top, text='GPS', variable=self._import_gps_var).pack(side='left', padx=(6, 0))
        ttk.Checkbutton(top, text='DAS', variable=self._import_das_var).pack(side='left', padx=(6, 0))

        self._import_btn = ttk.Button(top, text='Import', command=self._import)
        self._import_btn.pack(side='left', padx=(12, 0))
        ttk.Button(top, text='Close', command=self._close).pack(side='right')

        self.info = tk.StringVar(value='Pan: drag | Zoom: wheel | Select: click-drag rectangle | Min zoom: %d | Max segments: %d' % (MAP_MIN_ZOOM, MAP_MAX_SEGMENTS))
        ttk.Label(self, textvariable=self.info).pack(side='top', fill='x', padx=10)

        self.canvas = tk.Canvas(self, bg='#0f1115', highlightthickness=0)
        self.canvas.pack(side='top', fill='both', expand=True, padx=10, pady=10)
        self.canvas.bind('<Button-1>', self._on_down)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_up)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Button-4>', lambda e: self._zoom(1))
        self.canvas.bind('<Button-5>', lambda e: self._zoom(-1))

        self._pan_last = None
        self.after(50, self._redraw)

        # Pillow is required for tile rendering
        if Image is None or ImageTk is None:
            messagebox.showerror(
                'Map Import',
                'Missing dependency: Pillow (PIL).\n\nPlease install it and restart:\n  pip3 install pillow',
            )
            self.after(10, self._close)

    # --- tile math ---
    def _lon2x(self, lon: float, z: int) -> float:
        return (lon + 180.0) / 360.0 * (2 ** z)

    def _lat2y(self, lat: float, z: int) -> float:
        lat_rad = _deg2rad(lat)
        return (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * (2 ** z)

    def _x2lon(self, x: float, z: int) -> float:
        return x / (2 ** z) * 360.0 - 180.0

    def _y2lat(self, y: float, z: int) -> float:
        n = math.pi - 2.0 * math.pi * y / (2 ** z)
        return (180.0 / math.pi) * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

    def _tile_url(self, z: int, x: int, y: int) -> str:
        return f'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

    def _norm_tile_xy(self, z: int, x: int, y: int):
        """Normalize tile indices.

        - Wrap X around the dateline.
        - Clamp Y to valid range.
        """
        n = 2 ** z
        x = x % n
        y = max(0, min(n - 1, y))
        return x, y

    def _get_tile(self, z: int, x: int, y: int):
        """Return a tile image immediately, while loading missing tiles in the background.

        Tkinter must stay responsive, so we never do network I/O here.
        """
        x, y = self._norm_tile_xy(z, x, y)
        key = (z, x, y)

        # If already cached, return immediately.
        im = self._tile_cache.get(key)
        if im is not None:
            return im

        # Put a placeholder in cache immediately.
        if Image is None:
            raise RuntimeError('PIL is required for map tiles (Pillow missing).')

        placeholder = Image.new('RGBA', (256, 256), (40, 44, 52, 255))
        self._tile_cache[key] = placeholder

        # Request background load (deduplicated).
        if key not in self._tile_inflight and not self._tile_stop.is_set():
            self._tile_inflight.add(key)
            self._tile_req_q.put(key)

        return placeholder

    def _tile_loader_loop(self):
        """Background thread: fetch tiles and push results back to UI thread."""
        while not self._tile_stop.is_set():
            try:
                key = self._tile_req_q.get(timeout=0.25)
            except Empty:
                continue

            if key is None:
                continue

            z, x, y = key
            url = self._tile_url(z, x, y)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'SimStudio/0.46'})
                with urllib.request.urlopen(req, timeout=6) as r:
                    data = r.read()
                im = Image.open(BytesIO(data)).convert('RGBA')
                self._tile_res_q.put((key, im))
            except Exception:
                # Keep placeholder on any error.
                self._tile_res_q.put((key, None))
            finally:
                # allow future retries if needed
                try:
                    self._tile_inflight.remove(key)
                except KeyError:
                    pass
                try:
                    self._tile_req_q.task_done()
                except Exception:
                    pass

    def _process_loaded_tiles(self):
        """UI thread: apply loaded tiles and redraw with debounce."""
        any_updated = False
        while True:
            try:
                key, im = self._tile_res_q.get_nowait()
            except Empty:
                break

            if im is not None:
                self._tile_cache[key] = im
                # Invalidate the cached PhotoImage so _redraw() re-converts
                # the updated tile instead of showing the stale placeholder.
                self._tk_tile_cache.pop(key, None)
                any_updated = True

            try:
                self._tile_res_q.task_done()
            except Exception:
                pass

        if any_updated:
            self._schedule_redraw()

        # keep polling
        if not self._tile_stop.is_set():
            self.after(30, self._process_loaded_tiles)

    def _schedule_redraw(self):
        if self._tile_redraw_scheduled:
            return
        self._tile_redraw_scheduled = True

        def _do():
            self._tile_redraw_scheduled = False
            try:
                self._redraw()
            except Exception:
                pass

        self.after(40, _do)

    # --- interaction ---
    def _on_down(self, e):
        if self._select_mode:
            self._sel_start = (e.x, e.y)
            self._sel_end = (e.x, e.y)
        else:
            self._pan_last = (e.x, e.y)

    def _on_drag(self, e):
        if self._select_mode and self._sel_start:
            self._sel_end = (e.x, e.y)
            # Tiles are already cached — only the overlay rect changes, so
            # debouncing here costs nothing visually and prevents a full
            # PhotoImage rebuild on every mouse-move event.
            self._schedule_redraw()
            return
        if self._pan_last:
            dx = e.x - self._pan_last[0]
            dy = e.y - self._pan_last[1]
            self._pan_last = (e.x, e.y)
            # convert pixels to tile coords shift
            scale = 256
            cx = self._lon2x(self.center_lon, self.zoom)
            cy = self._lat2y(self.center_lat, self.zoom)
            cx -= dx / scale
            cy -= dy / scale
            self.center_lon = self._x2lon(cx, self.zoom)
            self.center_lat = self._y2lat(cy, self.zoom)
            self._schedule_redraw()

    def _on_up(self, e):
        self._pan_last = None
        if self._select_mode and self._sel_start and self._sel_end:
            # keep selection
            self._redraw()

    def _on_wheel(self, e):
        if e.delta > 0:
            self._zoom(+1)
        else:
            self._zoom(-1)

    def _zoom(self, dz: int):
        self.zoom = max(3, min(19, int(self.zoom + dz)))
        # Tiles at the new zoom level have different (z,x,y) keys, so the
        # previous zoom's PhotoImage cache is no longer needed.
        self._tk_tile_cache.clear()
        self._schedule_redraw()

    def _toggle_select(self):
        self._select_mode = not self._select_mode
        if not self._select_mode:
            self._sel_start = None
            self._sel_end = None
        self._redraw()

    def _search(self):
        q = (self.q.get() or '').strip()
        if not q:
            return
        try:
            url = 'https://nominatim.openstreetmap.org/search?' + urllib.parse.urlencode({
                'q': q,
                'format': 'json',
                'limit': 1,
            })
            req = urllib.request.Request(url, headers={'User-Agent': 'SimStudio/0.46'})
            with urllib.request.urlopen(req, timeout=12) as r:
                import json
                data = json.loads(r.read().decode('utf-8'))
            if not data:
                messagebox.showinfo('Search', 'No results.')
                return
            self.center_lat = float(data[0]['lat'])
            self.center_lon = float(data[0]['lon'])
            self._redraw()
        except Exception as ex:
            messagebox.showerror('Search', f'Failed: {ex}')

    def _bbox_latlon(self):
        if not (self._sel_start and self._sel_end):
            return None
        x0, y0 = self._sel_start
        x1, y1 = self._sel_end
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return None
        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)
        w = self.canvas.winfo_width() or 900
        h = self.canvas.winfo_height() or 600
        cx = self._lon2x(self.center_lon, self.zoom)
        cy = self._lat2y(self.center_lat, self.zoom)
        # pixel->tile fraction
        tx0 = cx + (left - w/2) / 256.0
        tx1 = cx + (right - w/2) / 256.0
        ty0 = cy + (top - h/2) / 256.0
        ty1 = cy + (bottom - h/2) / 256.0
        lon_w = self._x2lon(min(tx0, tx1), self.zoom)
        lon_e = self._x2lon(max(tx0, tx1), self.zoom)
        lat_n = self._y2lat(min(ty0, ty1), self.zoom)
        lat_s = self._y2lat(max(ty0, ty1), self.zoom)
        return (lat_s, lon_w, lat_n, lon_e)

    def _fetch_arcgis_points_geojson(self, layer_url: str, bbox: tuple[float, float, float, float]) -> list[dict]:
        """Fetch point features from an ArcGIS FeatureServer layer using a bbox.

        Args:
            layer_url: e.g. https://.../FeatureServer/0
            bbox: (lat_s, lon_w, lat_n, lon_e) in EPSG:4326

        Returns:
            List of GeoJSON features.
        """
        lat_s, lon_w, lat_n, lon_e = bbox
        # ArcGIS expects an envelope; we request geojson directly.
        params = {
            'f': 'geojson',
            'where': '1=1',
            'outFields': '*',
            'returnGeometry': 'true',
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'geometry': f"{lon_w},{lat_s},{lon_e},{lat_n}",
        }
        qurl = layer_url.rstrip('/') + '/query?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(qurl, headers={'User-Agent': 'SimStudio/0.56'})
        with urllib.request.urlopen(req, timeout=20) as r:
            import json
            js = json.loads(r.read().decode('utf-8'))
        return list(js.get('features', []) or [])

    def _import(self):
        # ------------------------------------------------------------------
        # Phase 1 — immediate validation and confirmation (main thread, fast)
        # ------------------------------------------------------------------
        if self.zoom < MAP_MIN_ZOOM:
            messagebox.showwarning('Zoom', f'Please zoom in more (min zoom: {MAP_MIN_ZOOM}).')
            return
        bbox = self._bbox_latlon()
        if bbox is None:
            messagebox.showwarning('Select Area', 'Please select an area first (Select Area → drag a rectangle).')
            return
        lat_s, lon_w, lat_n, lon_e = bbox

        # Confirm replace before starting the network request.
        has_existing = bool(self.app.world.segments)
        if has_existing:
            dlg = _ThreeOptionDialog(
                self.app,
                'Replace current scene?',
                'You already have a road network in the scene. What would you like to do?',
                'Save & Replace',
                'Replace (Don\'t Save)',
                'Cancel',
            )
            self.app.wait_window(dlg)
            if dlg.choice == 'opt1':
                self.app._save_scene()
            elif dlg.choice == 'opt3' or dlg.choice is None:
                return

        # Read toggle state now, before leaving the main thread.
        want_cameras = bool(self._import_cameras_var.get())

        # ------------------------------------------------------------------
        # Phase 2 — Overpass HTTP fetch on a background thread.
        # The UI stays responsive during the network wait.
        # ------------------------------------------------------------------
        self.info.set('Fetching roads from Overpass… (please wait)')
        self._import_btn.config(state='disabled')

        # Shared result container (written by background thread,
        # read by the after() callback on the main thread).
        _result: dict = {}

        _OVERPASS_MIRRORS = [
            'https://overpass-api.de/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter',
        ]

        def _do_fetch():
            import json as _json
            q = (f'[out:json][timeout:25];'
                 f'(way["highway"]({lat_s},{lon_w},{lat_n},{lon_e}););out geom;')
            js = None
            last_err = None  # plain assignment — Exception | None syntax requires Python 3.10+
            for _url in _OVERPASS_MIRRORS:
                try:
                    _data = urllib.parse.urlencode({'data': q}).encode('utf-8')
                    _req = urllib.request.Request(
                        _url, data=_data,
                        headers={'User-Agent': 'SimStudio/0.46'},
                    )
                    with urllib.request.urlopen(_req, timeout=30) as r:
                        js = _json.loads(r.read().decode('utf-8'))
                    break  # success — stop trying mirrors
                except Exception as e:
                    last_err = e
                    continue
            if js is None:
                _result['error'] = str(last_err)
                return

            _result['ways'] = [
                el for el in js.get('elements', [])
                if el.get('type') == 'way' and el.get('geometry')
            ]

            # Optional camera layer — also network I/O, so fetch here.
            cameras: list = []
            if want_cameras:
                try:
                    CAM_LAYER = ('https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/'
                                 'services/Traffic_Cameras_CDL/FeatureServer/0')
                    cameras = self._fetch_arcgis_points_geojson(
                        CAM_LAYER, (lat_s, lon_w, lat_n, lon_e)
                    )
                except Exception:
                    pass  # camera layer failure must not abort road import
            _result['cameras'] = cameras

        fetch_thread = threading.Thread(target=_do_fetch, daemon=True)
        fetch_thread.start()

        # ------------------------------------------------------------------
        # Phase 3 — poll for completion, then apply result (main thread).
        # ------------------------------------------------------------------
        _INFO_DEFAULT = (
            'Pan: drag | Zoom: wheel | Select: click-drag rectangle | '
            f'Min zoom: {MAP_MIN_ZOOM} | Max segments: {MAP_MAX_SEGMENTS}'
        )

        def _check_fetch():
            # If the map window was closed before the fetch finished, stop.
            # Tk does not cancel after() callbacks on destroy(), so we must
            # guard explicitly.  Without this, a late callback would overwrite
            # self.app.world after the user has already moved to a new scene.
            try:
                if not self.winfo_exists():
                    return
            except Exception:
                return

            if fetch_thread.is_alive():
                self.after(150, _check_fetch)
                return

            # Re-enable the button regardless of outcome.
            try:
                self._import_btn.config(state='normal')
                self.info.set(_INFO_DEFAULT)
            except Exception:
                pass

            if 'error' in _result:
                messagebox.showerror('Import', f'Overpass request failed: {_result["error"]}')
                return

            ways = _result.get('ways', [])
            seg_est = sum(max(0, len(w.get('geometry', [])) - 1) for w in ways)
            if seg_est > MAP_MAX_SEGMENTS:
                messagebox.showwarning(
                    'Too complex',
                    f'Selected area has ~{seg_est} road segments (> {MAP_MAX_SEGMENTS}). '
                    'Zoom in / select a smaller area.',
                )
                return

            # --- Build world from roads (fast, fine on main thread) ---
            lat0 = (lat_s + lat_n) / 2.0
            lon0 = (lon_w + lon_e) / 2.0

            def key_xy(x, y, q=0.5):
                return (round(x / q) * q, round(y / q) * q)

            nodes: dict = {}
            node_id_ctr = [1]

            def get_node_id(x, y):
                k = key_xy(x, y)
                if k in nodes:
                    return nodes[k]
                nid = f'n{node_id_ctr[0]}'
                node_id_ctr[0] += 1
                nodes[k] = nid
                return nid

            new_world = World()

            usage: dict = {}
            for w in ways:
                geom = w.get('geometry') or []
                for g in geom:
                    x, y = _latlon_to_xy_m(float(g['lat']), float(g['lon']), lat0, lon0)
                    k = key_xy(x, y)
                    usage[k] = usage.get(k, 0) + 1

            seg_id = 1
            for w in ways:
                geom = w.get('geometry') or []
                pts = []
                keys = []
                for g in geom:
                    x, y = _latlon_to_xy_m(float(g['lat']), float(g['lon']), lat0, lon0)
                    pts.append((x, y))
                    keys.append(key_xy(x, y))

                if len(pts) < 2:
                    continue

                start_i = 0
                for i in range(1, len(pts)):
                    is_end = (i == len(pts) - 1)
                    is_junction = is_end or (i == 0) or (usage.get(keys[i], 0) > 1)

                    x0, y0 = pts[i - 1]
                    x1, y1 = pts[i]
                    if math.hypot(x1 - x0, y1 - y0) > 120.0:
                        is_junction = True

                    if not is_junction:
                        continue

                    poly = pts[start_i:i + 1]
                    if len(poly) < 2:
                        start_i = i
                        continue
                    if polyline_length(poly) < 4.0:
                        start_i = i
                        continue

                    n0 = get_node_id(poly[0][0], poly[0][1])
                    n1 = get_node_id(poly[-1][0], poly[-1][1])
                    if n0 not in new_world.nodes:
                        new_world.nodes[n0] = Node(id=n0, x=poly[0][0], y=poly[0][1])
                    if n1 not in new_world.nodes:
                        new_world.nodes[n1] = Node(id=n1, x=poly[-1][0], y=poly[-1][1])

                    sid = f'seg{seg_id}'
                    seg_id += 1
                    new_world.segments[sid] = Segment(
                        id=sid, n0=n0, n1=n1,
                        lanes=1, lane_width=3.6, one_way=True, points=poly,
                    )
                    start_i = i

            # --- Add cameras from prefetched result ---
            cam_i = 1
            for f in _result.get('cameras', []):
                geom = (f or {}).get('geometry') or {}
                lon = geom.get('coordinates', [None, None])[0]
                lat = geom.get('coordinates', [None, None])[1]
                if lat is None or lon is None:
                    continue
                x, y = _latlon_to_xy_m(float(lat), float(lon), lat0, lon0)
                cid = f'cam{cam_i:03d}'
                cam_i += 1
                new_world.cameras[cid] = CameraSensor(
                    id=cid, x=float(x), y=float(y),
                    heading=0.0, fov_deg=70.0, range_m=250.0,
                )

            # GPS/DAS layer import is left as a stub (no stable public endpoint wired yet).

            # --- Apply to app ---
            self.app._stop(silent=True, skip_finalize=True)
            self.app.vp.reset()
            self.app.world = new_world
            self.app._normalize_single_lane()
            self.app.sim = Simulation(self.app.bus, self.app.world)
            self.app.sim.rebuild_lanes()
            self.app.selected = None
            self.app.pending = None
            self.app.undo = UndoRedo()
            self.app.undo.push(self.app.world)
            self.app._reset_runtime()

            deg = {nid: 0 for nid in self.app.world.nodes}
            for s in self.app.world.segments.values():
                deg[s.n0] = deg.get(s.n0, 0) + 1
                deg[s.n1] = deg.get(s.n1, 0) + 1
            self.app._open_end_nodes = {nid for nid, d in deg.items() if d <= 1}
            if self.app._open_end_nodes:
                messagebox.showwarning(
                    'Open Ends',
                    f'Imported roads contain {len(self.app._open_end_nodes)} open end(s). '
                    'Simulation can run, but some roads end without continuation.',
                )

            self.app._update_properties()
            self.app._render()
            self.app.status.set(
                f'Imported roads: {len(self.app.world.segments)} segments, '
                f'{len(self.app.world.nodes)} nodes'
            )
            self._close()

        self.after(150, _check_fetch)

    def _close(self):
        # Stop background tile loader cleanly.
        try:
            self._tile_stop.set()
            self._tile_req_q.put(None)
        except Exception:
            pass

        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _redraw(self):
        self.canvas.delete('all')
        self._img_refs = []
        w = self.canvas.winfo_width() or 900
        h = self.canvas.winfo_height() or 600

        if ImageTk is None:
            return

        cx = self._lon2x(self.center_lon, self.zoom)
        cy = self._lat2y(self.center_lat, self.zoom)
        # top-left tile fractional coords
        tx0 = cx - (w / 2) / 256.0
        ty0 = cy - (h / 2) / 256.0

        x_start = int(math.floor(tx0))
        y_start = int(math.floor(ty0))
        x_end = int(math.floor(cx + (w / 2) / 256.0))
        y_end = int(math.floor(cy + (h / 2) / 256.0))

        for xt in range(x_start, x_end + 1):
            for yt in range(y_start, y_end + 1):
                px = int((xt - tx0) * 256)
                py = int((yt - ty0) * 256)
                try:
                    im = self._get_tile(self.zoom, xt, yt)
                    # Key must match _tile_cache / _process_loaded_tiles: (z, x, y)
                    xn, yn = self._norm_tile_xy(self.zoom, xt, yt)
                    key = (self.zoom, xn, yn)
                    tk_im = self._tk_tile_cache.get(key)
                    if tk_im is None:
                        tk_im = ImageTk.PhotoImage(im)
                        self._tk_tile_cache[key] = tk_im
                except Exception:
                    # In case of network errors, keep the canvas responsive.
                    continue
                self._img_refs.append(tk_im)
                self.canvas.create_image(px, py, image=tk_im, anchor='nw')

        # selection rectangle
        if self._sel_start and self._sel_end:
            x0, y0 = self._sel_start
            x1, y1 = self._sel_end
            self.canvas.create_rectangle(x0, y0, x1, y1, outline='#ffd166', width=2, dash=(5, 3))

        # crosshair
        self.canvas.create_line(w/2 - 8, h/2, w/2 + 8, h/2, fill='#ffffff')
        self.canvas.create_line(w/2, h/2 - 8, w/2, h/2 + 8, fill='#ffffff')
        self.canvas.create_text(10, 10, anchor='nw', fill='#ffffff', text=f'Zoom {self.zoom} | {self.center_lat:.5f}, {self.center_lon:.5f}')



def _set_macos_process_name(name: str) -> None:
    """Set the macOS Dock/tooltip name instead of 'Python'.

    Three layers are needed for the name to appear correctly:
    1. NSProcessInfo.setProcessName  — affects Activity Monitor & menu bar
    2. CFBundleName in the main bundle info dict — affects Dock tooltip
    3. pyobjc NSBundle fallback — works if pyobjc is installed
    """
    encoded = name.encode("utf-8")

    # --- Layer 1 & 2: via Objective-C runtime (no pyobjc needed) ---
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.CDLL(ctypes.util.find_library("objc"), use_errno=True)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        def _nsstr(s: bytes) -> ctypes.c_void_p:
            NSString = objc.objc_getClass(b"NSString")
            sel = objc.sel_registerName(b"stringWithUTF8String:")
            return objc.objc_msgSend(NSString, sel, s)

        name_ns = _nsstr(encoded)

        # 1. NSProcessInfo.setProcessName:
        NSProcessInfo = objc.objc_getClass(b"NSProcessInfo")
        info = objc.objc_msgSend(NSProcessInfo, objc.sel_registerName(b"processInfo"))
        objc.objc_msgSend(info, objc.sel_registerName(b"setProcessName:"), name_ns)

        # 2. Set CFBundleName and CFBundleDisplayName in the main bundle dict
        NSBundle = objc.objc_getClass(b"NSBundle")
        bundle = objc.objc_msgSend(NSBundle, objc.sel_registerName(b"mainBundle"))
        info_dict = objc.objc_msgSend(bundle, objc.sel_registerName(b"infoDictionary"))
        set_sel = objc.sel_registerName(b"setObject:forKey:")
        objc.objc_msgSend(info_dict, set_sel, name_ns, _nsstr(b"CFBundleName"))
        objc.objc_msgSend(info_dict, set_sel, name_ns, _nsstr(b"CFBundleDisplayName"))
    except Exception:
        pass

    # --- Layer 3: pyobjc (if available, most reliable) ---
    try:
        from Foundation import NSBundle as _NSBundle  # type: ignore
        info = _NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = name
        info["CFBundleDisplayName"] = name
    except Exception:
        pass


def run():
    _set_macos_process_name("Optical Fiber SIM")
    app = SimStudioApp()
    app.mainloop()