import json
from typing import Dict, Any
from .models import World, Node, Segment, LaneGeom, Vehicle, GPSSensor, CameraSensor, DASSensor

def world_to_dict(w: World) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "nodes": {k: {"x": n.x, "y": n.y} for k,n in w.nodes.items()},
        "segments": {k: s.__dict__ for k,s in w.segments.items()},
        "lanes": {k: {"id": l.id, "segment_id": l.segment_id, "offset_index": l.offset_index, "polyline": l.polyline} for k,l in w.lanes.items()},
        "vehicles": {k: v.__dict__ for k,v in w.vehicles.items()},
        "gps": {k: g.__dict__ for k,g in w.gps.items()},
        "cameras": {k: c.__dict__ for k,c in w.cameras.items()},
        "das": {k: d.__dict__ for k,d in w.das.items()},
    }
    if not getattr(w, "auto_spawn", True):
        d["auto_spawn"] = False
    if getattr(w, "boundary_exit_nodes", None):
        d["boundary_exit_nodes"] = sorted(w.boundary_exit_nodes)
    return d

def dict_to_world(d: Dict[str, Any]) -> World:
    w = World()
    for k,v in d.get("nodes", {}).items():
        w.nodes[k] = Node(id=k, x=float(v["x"]), y=float(v["y"]))
    for k,v in d.get("segments", {}).items():
        vv = dict(v)
        if "points" in vv and vv["points"] is not None:
            vv["points"] = [tuple(p) for p in vv.get("points", [])]
        w.segments[k] = Segment(**vv)
    for k,v in d.get("lanes", {}).items():
        w.lanes[k] = LaneGeom(id=v["id"], segment_id=v["segment_id"], offset_index=int(v["offset_index"]), polyline=[tuple(p) for p in v["polyline"]])
    for k,v in d.get("vehicles", {}).items():
        vv = dict(v)
        # Backward compatibility: earlier versions had footprint fields.
        vv.pop("length_m", None)
        vv.pop("width_m", None)
        w.vehicles[k] = Vehicle(**vv)
    for k,v in d.get("gps", {}).items():
        w.gps[k] = GPSSensor(**v)
    for k,v in d.get("cameras", {}).items():
        w.cameras[k] = CameraSensor(**v)
    for k,v in d.get("das", {}).items():
        vv = dict(v)
        # Backward compatibility: earlier versions had detection/merging fields.
        vv.pop("snr_det_threshold", None)
        vv.pop("bias_k", None)
        vv.pop("peak_merge_threshold_m", None)
        w.das[k] = DASSensor(**vv)
    # Optional top-level flag: set to false to disable automatic vehicle spawning.
    if "auto_spawn" in d:
        w.auto_spawn = bool(d["auto_spawn"])
    # Optional boundary exit nodes — roads that leave the map tile.
    # Vehicles reaching these nodes exit cleanly (route_complete, not stuck).
    if "boundary_exit_nodes" in d:
        w.boundary_exit_nodes = set(str(n) for n in d["boundary_exit_nodes"])
    return w

def save_world(path, w: World) -> None:
    path.write_text(json.dumps(world_to_dict(w), indent=2))

def load_world(path) -> World:
    return dict_to_world(json.loads(path.read_text()))
