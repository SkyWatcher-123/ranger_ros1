# ranger_shuttle.py

This node sends the Ranger Mini back and forth between two X positions in the
map frame, 1.2 m and 4.8 m by default. It logs every stop caused by a person,
every other stop, every leg and the trajectory, so you can estimate how often
people interrupt the robot and what that costs.

The node does not drive the base. It only calls `go_to_base_x.py`
(`/ranger/set_base_x`, in this package) with A and B as alternating targets.
Line following, the 0.1 m/s speed and the human pause work exactly as they do
without this node.

## Run

Start each of these in its own terminal:

```bash
roslaunch ranger_bringup apriltag_to_ranger_joint_state.launch
```

```bash
roslaunch ranger_bringup line_following.launch
```

```bash
roslaunch tracking_3d human_3d_tracker.launch target_frame:=map
```

```bash
roslaunch ranger_bringup ranger_shuttle.launch x_a:=1.2 x_b:=4.8 run_tag:=hallway_busy
```

The tracker is optional. Without it, pauses are still counted, but the
positions of people are not logged.

Press Ctrl-C in the shuttle terminal to stop. The node calls
`/clear_ranger_goal`, so the base stops at once, then writes the final
`summary.json`. Start the shuttle in its own launch, not inside
`line_following.launch`. Otherwise Ctrl-C also kills `go_to_base_x` before the
shuttle can cancel the goal.

The run can also end on its own:

- `max_travels:=N` ends the run when the robot arrives after the Nth travel.
- `max_duration:=S` ends the run at the first arrival after S seconds.
- An external `rosservice call /clear_ranger_goal` ends the run when
  `on_cancel:=stop` (the default). With `on_cancel:=resume`, the same target is
  sent again after 5 s.
- A leg that spends too long without arriving ends the run and cancels the goal.
  Time paused for a person doesn't count toward this limit. The default is
  3 × span / speed + 30 s.

If the shuttle is killed with `kill -9`, `go_to_base_x` still finishes the
current leg, then stops.

### Where the robot starts

| Start position | First goal | Counted as a travel? |
|---|---|---|
| Within 0.10 m of A | B | yes |
| Within 0.10 m of B | A | yes |
| Anywhere else | the nearer end | no (approach leg) |

`first_target:=a` or `first_target:=b` forces the first goal.

## What is counted

- **Human intervention:** each time `go_to_base_x` switches to `PAUSED_HUMAN`.
  This happens when a person is ahead in the travel direction, within
  `human_x_range` along x and `human_y_range` sideways of `base_link`. The pause
  ends when the status changes again. Usually that is `MOVING`, after the path
  has been clear for `human_resume_delay`.
  - `duration` is the time from stop to resume.
  - `blocked = duration − human_resume_delay` is the time the path was actually
    blocked. It is exact for pauses that ended with `RESUMED`.
- **Stall:** the status is `MOVING`, but the base moved less than 3 cm in 2 s.
  Typical causes are tape lost (`line_keeper`), the heading being refused, or a
  wheel slipping. Stalls are logged apart from human stops. The start of a stall
  is set back to the start of that 2 s window.
- **Travel:** the robot arrives at one end after it last arrived at the other.
  The approach leg is not a travel. A leg that was cancelled and sent again
  still counts as one travel when it arrives. `round_trips = travels // 2`.
- **Distance:** measured from the TF pose with a 2 cm step, so the pose noise of
  a standing robot is not added. Each leg stops when it enters the arrival
  tolerance, so a travel is about `span − 2 × tolerance` long. The summary gives
  both `distance_m` (measured) and `nominal_distance_m` (travels × span).

The person-ahead fields (`h_*`) use the same rule as `go_to_base_x`. The node
reads that rule's parameters from `/go_to_base_x/*`.

## Output

Each run gets a folder: `~/ranger_shuttle_logs/<YYYYmmdd_HHMMSS>[_tag]/`.

`t` is ROS time in seconds. `t_rel` is the time since the run started. `wall`
is `t` as local ISO time. `dir` is the direction of travel along map x: +1 or
−1. All CSV columns are listed in `scripts/ranger_shuttle.py` (`*_HDR`).

| File | One row per | Main columns |
|---|---|---|
| `stops.csv` | stop | `cause` (human or stall), `leg`, `to_end`, `t_start`, `wall_start`, `t_end`, `duration`, `blocked`, `end_reason` (RESUMED, REACHED, ABORTED, …), `x_start`, `y_start`, `x_end`, `drift` (creep while stopped), `leg_progress` (0 to 1), `dist_to_target`, `vx_onset`, `h_ahead` and `h_lat` (person in `base_link` at the stop), `h_map_x` and `h_map_y` (person in the map) |
| `legs.csv` | goal sent | `from_end`, `to_end`, `duration`, `result` (REACHED, CANCELLED, LOST, ABORTED, REJECTED), `completes_travel`, `travel_no`, `final_error`, `path_len`, `n_human`, `human_time`, `n_stall`, `stall_time`, `moving_time` |
| `events.csv` | state change | RUN_START, LEG_START, GOAL_SENT, STOP_START, STOP_END, LEG_END, POSE_LOST, POSE_OK, FINISH_AFTER_LEG, LEG_TIMEOUT, CANCEL_RESUME, END_REQUEST, RUN_END |
| `trajectory.csv` | sample (10 Hz) | `status`, `stop`, `x`, `y`, `yaw`, `vx`, `odom`, `h_n` (people seen), `h_nearest`, `h_ahead`, `h_lat`, `h_block` (person inside the pause zone) |
| `summary.json` | run | see below |
| `config.json` | run | parameters of this node and of `go_to_base_x`, host, start time |

`summary.json` is rewritten every 10 s, so a crash still leaves recent totals.
It contains:

- `travel`: travels, round trips, legs by result, distance, travel time (gross
  and net of stops), arrival error.
- `time_s`: `moving`, `human`, `stall` and `idle` time. Idle is the dwell at
  each end plus start-up. The four add up to `run_duration_s`.
- `human_interventions`:
  - the count, and how many legs had one;
  - the count per travel, per 100 m, per hour of run time and per hour of
    moving;
  - the mean distance and the mean moving time between interventions;
  - `duration_s`, `blocked_s`, `person_ahead_at_onset_m` and
    `drift_while_stopped_m`, each with n, mean, std, min, p50, p90 and max;
  - the count by destination end;
  - a histogram of where along x the pauses happened, by direction.
- `stalls`: count and duration statistics.

About 3 MB per hour at 10 Hz, nearly all of it `trajectory.csv`. Use `traj_rate:=2` for 0.6 MB per hour.

## Analysis starting point

```python
import json, pandas as pd
from scipy.stats import chi2

run = "~/ranger_shuttle_logs/20261103_140000_hallway_busy/"
s = json.load(open(run + "summary.json"))
stops = pd.read_csv(run + "stops.csv")
h = stops[stops.cause == "human"]

n, km = len(h), s["travel"]["distance_m"] / 1000
lo = chi2.ppf(0.025, 2 * n) / 2 / km if n else 0.0          # exact Poisson 95% CI
hi = chi2.ppf(0.975, 2 * n + 2) / 2 / km
print(f"{n} interventions over {km * 1000:.0f} m: {n / km:.1f}/km [{lo:.1f}, {hi:.1f}]")
print(h[["duration", "blocked"]].describe())
print(h.groupby((h.x_start // 0.2) * 0.2).size())         # pauses per 0.2 m of track
```

To pool several runs, concatenate their `stops.csv` and add up their exposure:
`distance_m`, `time_s.moving` or `travels_completed`.

## Parameters

These are private parameters (`~name`). The launch file exposes the common
ones as args.

| Name | Default | |
|---|---|---|
| `x_a`, `x_b` | 1.2, 4.8 | m, the two ends (map x) |
| `dwell` | 1.0 | s to wait at each end |
| `tolerance` | 0 | m arrival tolerance sent with each goal; 0 = `go_to_base_x` default (0.03) |
| `max_travels` / `max_duration` | 0 / 0 | end conditions; 0 = no limit |
| `finish_leg_on_limit` | true | at `max_duration`, finish the current leg first |
| `first_target` | auto | auto, a or b |
| `start_tolerance` | 0.10 | m, "already at an end" at start |
| `on_cancel`, `cancel_resume_delay` | stop, 5.0 | what to do after an external `/clear_ranger_goal` |
| `leg_timeout` | 0 (auto) | s of moving or stalled time per leg before the run is aborted |
| `stall_window`, `stall_min_dist` | 2.0, 0.03 | stall detector; `stall_window: 0` turns it off |
| `traj_rate` | 10 | Hz, `trajectory.csv` |
| `summary_period` | 10 | s, `summary.json` rewrite and console line |
| `bin_size` | 0.2 | m, histogram bins for pause locations |
| `odom_step` | 0.02 | m, distance hysteresis |
| `log_dir`, `run_tag` | `~/ranger_shuttle_logs`, "" | output folder |
| `human_log` | true | subscribe to the people topic to log positions |
| `go_to_base_x_ns` | `/go_to_base_x` | where `speed` and `human_*` are read from; `~human_x_range` and similar override them |
| `set_base_x_service`, `clear_goal_service`, `status_topic` | `/ranger/set_base_x`, `/clear_ranger_goal`, `/ranger/base_goal_status` | interfaces of `line_following.launch` |
| `map_frame`, `base_frame` | map, base_link | pose source (TF) |
