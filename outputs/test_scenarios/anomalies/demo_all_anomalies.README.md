# `demo_all_anomalies.sim.json` — visual anomaly demo

One scenario that exercises every anomaly class in a single 30-second run.
Load it from the GUI ("Open Scene") just like any other `.sim.json` file.

## Layout

Four parallel one-way roads, 800 m long, stacked top-to-bottom 20 m apart:

| Road                  | y     | Lane                       | Purpose                              |
|-----------------------|-------|----------------------------|--------------------------------------|
| `Snorth`              | +30   | `Snorth_fwd_lane1`         | weaving (lateral anomaly)            |
| `Smiddle`             | +10   | `Smiddle_fwd_lane1`        | obstacle + IDM braking chain         |
| `Ssouth`              | -10   | `Ssouth_fwd_lane1`         | collision pair                        |
| `Sfarsouth`           | -30   | `Sfarsouth_fwd_lane1`      | route-complete + normal cruiser      |

A single GPS sensor at the road centre and a single Camera looking across all
four lanes guarantee the existing sensor pipeline keeps publishing events
during the demo. A DAS fiber runs along the middle road so you can see the
DAS waterfall pick up the obstacle and the stopping STOPPER.

## Vehicles

| Id              | Lane                    | Demonstrates                                              |
|-----------------|-------------------------|-----------------------------------------------------------|
| `WEAVER`        | Snorth (weave lane)     | `lateral_mode="weave"` — sinusoidal ±2.5 m at 4 s period  |
| `OBSTACLE_PED`  | Smiddle, s=300 (frozen) | Road obstacle — drawn as a yellow warning diamond         |
| `STOPPER`       | Smiddle, s=80, v=22.2   | Approaches OBSTACLE_PED and brakes via IDM                |
| `TAILBACK`      | Smiddle, s=20, v=22.2   | Chain-brakes behind STOPPER (no special marker)           |
| `WALL`          | Ssouth, s=160 (frozen)  | Stalled car the CRASHER will hit                          |
| `CRASHER`       | Ssouth, s=20, v=22.2    | `allow_collision=True`, `disable_idm=True` — drives into WALL |
| `ROUTE_DEMO`    | Sfarsouth, s=80, v=10   | Has a planned route to s=180 — gains the **blue X**        |
| `CRUISER`       | Sfarsouth, s=300, v=15  | Normal vehicle, no anomaly — visual baseline               |

## Timeline (what to watch and when)

| Time   | Event                                                                                  |
|-------:|----------------------------------------------------------------------------------------|
| t = 0  | Press **Start**. `WEAVER` immediately begins oscillating across its lane (north road). |
| t ≈ 6.15 s  | `CRASHER` rear-ends `WALL` on the south road. **One** `world.collision` event fires; **both** vehicles get a red warning triangle and are frozen at impact. |
| t ≈ 10.0 s | `STOPPER` on the middle road is fully stopped just behind `OBSTACLE_PED` (gap ≈ 2 m). The pedestrian remains drawn as a yellow diamond. |
| t ≈ 10.05 s | `ROUTE_DEMO` reaches its planned waypoint at s=180. `world.route_complete` fires; a **blue X** appears on the vehicle. |
| t ≈ 12.0 s | `TAILBACK` is fully stopped behind `STOPPER`. No marker — it's a normal vehicle that happens to be stopped (still acts as a physical IDM blocker for anything behind it). |
| 12 → 30 s   | `CRUISER` keeps driving normally; `WEAVER` keeps weaving; the four markers stay on screen. |

## Visual legend

- **Yellow diamond with an exclamation mark** — Road obstacle. Set on a `Vehicle` whose `object_type` is anything other than `"vehicle"` (e.g. `"pedestrian"`, `"debris"`, `"animal"`, `"barrier"`). Always paired with `frozen=True`.
- **Red warning triangle** — Collision marker, drawn on **both** vehicles in the colliding pair. The same triangle on the same colour confirms they're part of the same accident. Each colliding pair publishes exactly one `world.collision` event.
- **Blue X** — Route completed normally (`world.route_complete`). Does **not** mean the vehicle is dead-ended. The red X you may also see on dead-end nodes is for `world.vehicle_stuck` and is unchanged from before this demo.
- **Blue rectangle** (darker = heavier) — Normal vehicle. A vehicle that simply stopped in the middle of the road keeps this appearance — there's no special marker for "stopped" because the user asked for it not to be confused with route completion or a crash.

## What to verify visually

1. **Weaving** — `WEAVER` (top road) clearly crosses both lane edges.
2. **Yellow obstacle** — `OBSTACLE_PED` on the middle road is unmistakably an obstacle, not a regular vehicle.
3. **Physical blocking** — `STOPPER` then `TAILBACK` come to a halt with a realistic gap behind the obstacle, with no special markers on either of them.
4. **Collision pair** — `CRASHER` and `WALL` both show the red warning triangle, both are frozen at the impact location.
5. **Route completion** — `ROUTE_DEMO` shows the blue X around the time it reaches s=180, and continues to display it for the rest of the run.
6. **Baseline** — `CRUISER` looks like any vehicle in any other scenario: a dark blue rectangle, no markers.

## Re-creating / tweaking the demo

The scenario is authored by hand in `demo_all_anomalies.sim.json`. To add or
remove vehicles, edit the JSON directly and reload. Headless verification:

```bash
PYTHONPATH=src python3 - << 'EOF'
from pathlib import Path
from simstudio.bus import EventBus
from simstudio.project_io import load_world
from simstudio.sim_core import Simulation
w = load_world(Path("outputs/test_scenarios/anomalies/demo_all_anomalies.sim.json"))
sim = Simulation(EventBus(), w); sim.rebuild_lanes()
sim.set_route_to_point("ROUTE_DEMO", "Sfarsouth_fwd_lane1", 180.0)
events = []
for t in ("world.collision", "world.route_complete"):
    sim.bus.subscribe(t, lambda ev: events.append((ev.topic, ev.payload)))
for _ in range(int(30/0.05)):
    sim.step(0.05)
for t, p in events:
    print(t, p)
EOF
```

You should see exactly one `world.collision` (CRASHER + WALL) and exactly one
`world.route_complete` (ROUTE_DEMO) event.
