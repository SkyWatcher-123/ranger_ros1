# ranger_navigation

Autonomous navigation for the AgileX **Ranger Mini 3.0**, modelled directly on
[`husky_navigation`](https://github.com/husky/husky/tree/noetic-devel/husky_navigation).
This package provides the `move_base` pipeline — a **global planner**
(`navfn/NavfnROS`) and a **local planner** (`dwa_local_planner/DWAPlannerROS`)
running over layered costmaps — adapted to the Ranger Mini's size and
kinematics.

It is the equivalent of Husky's `move_base.launch`. It does **not** include
gmapping or amcl; those come later (the package already declares the `amcl`
dependency so adding it is just a launch file).

## Files (and the Husky file each mirrors)

| ranger_navigation | husky_navigation |
|---|---|
| `launch/ranger_mini_move_base.launch` | `launch/move_base.launch` |
| `launch/move_base_mapless_demo.launch` | `launch/move_base_mapless_demo.launch` |
| `launch/ranger_mini_move_base_demo.launch` | `launch/amcl_demo.launch` (but with a static `map→odom` instead of amcl) |
| `config/costmap_common.yaml` | same |
| `config/costmap_local.yaml` | same |
| `config/costmap_global_static.yaml` | same |
| `config/costmap_global_laser.yaml` | same |
| `config/planner.yaml` | same |

## What had to change from Husky (and why)

Husky is a skid-steer **differential-drive** robot. The Ranger Mini is a
four-wheel-independent-steer robot, but `ranger_base` already exposes a plain
`geometry_msgs/Twist` interface on `/cmd_vel`, so we drive it the same
diff-drive way Husky does. The differences are:

1. **Footprint / inflation** — Ranger Mini ≈ 0.74 m × 0.50 m (wheelbase
   0.494 m, track 0.364 m) vs Husky's 1.0 m × 0.66 m, so the footprint is
   smaller and `inflation_radius` was reduced from 1.0 m to 0.4 m.
2. **Velocity / acceleration limits** — defaults kept conservative
   (`max_vel_x` 0.5 m/s, `max_vel_rot` 1.0 rad/s; RM3 hardware can do ~1.5 m/s
   and ~4.8 rad/s). `acc_lim` lowered from Husky's 2.5/3.2 to 1.0/2.0 for the
   smaller, car-like base.
3. **I/O wiring** — `move_base` publishes to `/cmd_vel` and reads `/odom`
   directly (Husky routes through `twist_mux`; the Ranger base subscribes to
   `/cmd_vel` itself, so no mux is needed).

### Kinematic caveat — minimum turn radius

`ranger_base` maps `/cmd_vel` to a motion mode based on the requested turn
radius `r = linear.x / angular.z`:

* `r ≥ min_turn_radius` (~0.48 m) → **dual-Ackermann** (smooth car-like arc)
* `r < min_turn_radius` → **spinning** (rotates in place, `linear.x` ignored)
* `linear.y ≠ 0` → **parallel / crab** (sideways)

DWA is configured as diff-drive (`y` disabled), so it only emits `linear.x` +
`angular.z`. The base then does Ackermann for gentle arcs and spinning for
tight turns / rotate-to-goal, so the robot reaches any pose. The trade-off:
when DWA asks for a tight arc (low `x`, high `yaw`) the base **spins in place**
instead of curving — motion near sharp turns is "stop-rotate-go" rather than
smooth. This is good enough to bring the pipeline up and matches Husky's
behaviour closely. For genuinely smooth Ackermann arcs later, switch the local
planner to `teb_local_planner` with `min_turning_radius: 0.48` and
`max_vel_y: 0`.

## Prerequisites to actually drive the robot

`move_base` needs a complete TF tree `map → odom → base_link` plus a map.
Three things must be running in addition to this package:

1. **Base driver with odom TF.** The Ranger base publishes `/odom` but, by
   default, **not** the `odom → base_link` transform. Launch the bringup with
   it enabled:
   ```bash
   roslaunch ranger_bringup ranger_mini_v3.launch publish_odom_tf:=true
   ```
   (The `ranger_mini_v3.launch` added alongside this work defaults
   `publish_odom_tf` to `true`.)
2. **A map** published by `map_server` (your provided `.yaml`/`.pgm`).
3. **A `map → odom` transform.** Normally amcl provides this. Until you add
   amcl, use a static identity transform (the robot then navigates on
   odometry alone, assuming it starts at the map origin).

Items 2 and 3 are wired up for you in `ranger_mini_move_base_demo.launch`.

## Quick start (no sensors, no amcl, provided map)

Terminal 1 — base driver (with odom TF):
```bash
roslaunch ranger_bringup ranger_mini_v3.launch publish_odom_tf:=true
```

Terminal 2 — map server + static `map→odom` + move_base:
```bash
roslaunch ranger_navigation ranger_mini_move_base_demo.launch \
    map_file:=/path/to/your_map.yaml
```
(Omit `map_file` to use the bundled empty 5×5 m map for a smoke test.)

Terminal 3 — visualise and send goals:
```bash
rosrun rviz rviz
```
Add a Map, Global Plan, Local Plan and TF display, set the fixed frame to
`map`, then use **2D Nav Goal** to command the robot.

> Without a LiDAR the local costmap has no obstacle data, so the robot follows
> the global plan computed from the static map but cannot react to unmapped or
> moving obstacles. That is expected for this sensor-less stage.

## Just the planner (exact Husky `move_base.launch` analogue)

If you are wiring the map/TF up yourself:
```bash
roslaunch ranger_navigation ranger_mini_move_base.launch
```
Useful args: `no_static_map` (default `false`), `base_global_planner`,
`base_local_planner`, `cmd_vel_topic` (default `/cmd_vel`), `odom_topic`
(default `/odom`).

## Next steps

* **Add a LiDAR** → set the `laser` `topic` in `config/costmap_common.yaml`
  and publish a `base_link → <laser_frame>` static transform. The obstacle
  layers come alive immediately.
* **Add amcl** → drop in an `amcl.launch` (like Husky's) and replace the
  static `map→odom` in the demo with it for real localisation.
* **Add gmapping** → for building the map in the first place.
