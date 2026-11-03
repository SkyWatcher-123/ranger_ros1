#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shuttle the Ranger Mini back and forth between two X positions and log every stop.

Once started, the node sends go_to_base_x goals (ranger_msgs/SetBaseX on
/ranger/set_base_x) that alternate between ~x_a and ~x_b (map frame, default
1.2 m and 4.8 m), with ~dwell seconds at each end. It does not drive the base
itself: go_to_base_x, line_keeper and the human pause work exactly as before.

What is recorded, one folder per run (~log_dir/<YYYYmmdd_HHMMSS>[_tag]/):

  config.json      parameters of this node and of go_to_base_x, start time.
  events.csv       every state change: RUN_START, LEG_START, STOP_START,
                   STOP_END, LEG_END, POSE_LOST/POSE_OK, RUN_END, ...
  stops.csv        one row per stop. cause=human: go_to_base_x reported
                   PAUSED_HUMAN (a person in the path). cause=stall: status
                   MOVING but the base did not move (tape lost, heading
                   refused, ...). Where, when, how long, what was in front.
  legs.csv         one row per goal (one leg): result, duration, stops.
  trajectory.csv   pose, speed, go_to_base_x status and the nearest person,
                   at ~traj_rate Hz.
  summary.json     totals and statistics, rewritten every ~summary_period
                   seconds and at the end, so a crash still leaves one.

Counting:
  * travel = arriving at one end after having last arrived at the other one.
    The first leg from a position between the ends ("approach") is not a
    travel. A leg that was cancelled and re-sent still completes a travel
    when it arrives. round_trips = travels // 2.
  * human intervention = one PAUSED_HUMAN onset. A pause lasts until
    go_to_base_x resumes, so it includes its ~human_resume_delay;
    blocked = duration - human_resume_delay is the time the path was blocked.

Stopping: Ctrl-C (or SIGTERM) cancels the active goal through
/clear_ranger_goal, so the base stops at once, then writes the summary.
~max_travels / ~max_duration end the run at the next arrival.
An external /clear_ranger_goal ends the run too (~on_cancel: stop), or the
same goal is re-sent after ~cancel_resume_delay seconds (~on_cancel: resume).
"""

import csv
import json
import math
import os
import signal
import socket
import threading
import time
from collections import deque
from datetime import datetime

import rospy
import tf2_ros
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ranger_msgs.srv import SetBaseX


# ------------------------------- small helpers ---------------------------------
def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def transform_point(tr, p):
    """Apply a geometry_msgs/TransformStamped to a point p=(x, y, z)."""
    q, t = tr.transform.rotation, tr.transform.translation
    x, y, z = p
    tx = 2.0 * (q.y * z - q.z * y)
    ty = 2.0 * (q.z * x - q.x * z)
    tz = 2.0 * (q.x * y - q.y * x)
    return (x + q.w * tx + (q.y * tz - q.z * ty) + t.x,
            y + q.w * ty + (q.z * tx - q.x * tz) + t.y,
            z + q.w * tz + (q.x * ty - q.y * tx) + t.z)


def detection_score(results):
    """Best hypothesis score (vision_msgs ROS1: r.score; newer: r.hypothesis.score)."""
    scores = []
    for r in results:
        s = getattr(r, "score", None)
        if s is None and hasattr(r, "hypothesis"):
            s = r.hypothesis.score
        if s is not None:
            scores.append(s)
    return max(scores) if scores else 1.0


def wall_iso(t=None):
    """Local ISO time of a ROS time stamp [s] (wall clock on the robot, bag time in sim)."""
    return datetime.fromtimestamp(time.time() if t is None else t).isoformat(timespec="milliseconds")


def rnd(v, n=3):
    return None if v is None else round(float(v), n)


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def describe(vals):
    """n, total, mean, std, min, p50, p90, max of a list of numbers."""
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    s = sorted(vals)
    return {"n": n, "total": rnd(sum(vals)), "mean": rnd(mean), "std": rnd(std),
            "min": rnd(s[0]), "p50": rnd(percentile(s, 50)), "p90": rnd(percentile(s, 90)),
            "max": rnd(s[-1])}


def ratio(a, b, n=4):
    return rnd(a / b, n) if b else None


class CsvLog(object):
    def __init__(self, path, header):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(header)
        self.f.flush()

    @staticmethod
    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float):
            return "%.4f" % v if abs(v) < 1e5 else "%.3f" % v
        return v

    def row(self, values, flush=True):
        self.w.writerow([self._fmt(v) for v in values])
        if flush:
            self.f.flush()

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


TRAJ_HDR = ["t", "t_rel", "leg", "target_x", "dir", "status", "stop",
            "x", "y", "yaw", "vx", "odom",
            "h_n", "h_nearest", "h_ahead", "h_lat", "h_block"]
EVENT_HDR = ["t", "t_rel", "wall", "event", "leg", "target_x", "x", "y", "detail"]
STOP_HDR = ["stop_id", "cause", "leg", "to_end", "dir", "target_x",
            "t_start", "t_rel_start", "wall_start", "t_end", "wall_end",
            "duration", "blocked", "end_reason",
            "x_start", "y_start", "yaw_start", "x_end", "drift",
            "leg_progress", "dist_from_leg_start", "dist_to_target", "vx_onset",
            "h_ahead", "h_lat", "h_map_x", "h_map_y", "h_n"]
LEG_HDR = ["leg", "from_end", "to_end", "from_x", "target_x", "dir",
           "t_start", "wall_start", "t_end", "duration", "result",
           "completes_travel", "travel_no", "x_end", "final_error", "path_len",
           "n_human", "human_time", "n_stall", "stall_time", "moving_time"]


# ---------------------------------- the node -----------------------------------
class Shuttle(object):
    def __init__(self):
        p = rospy.get_param
        self.x = {"A": float(p("~x_a", 1.2)), "B": float(p("~x_b", 4.8))}
        self.span = abs(self.x["B"] - self.x["A"])
        if self.span < 0.05:
            raise ValueError("x_a and x_b are %.3f m apart; need at least 0.05 m" % self.span)
        self.tolerance = float(p("~tolerance", 0.0))          # <= 0: go_to_base_x default
        self.dwell = max(0.0, float(p("~dwell", 1.0)))
        self.max_travels = int(p("~max_travels", 0))           # 0 = no limit
        self.max_duration = float(p("~max_duration", 0.0))     # s, 0 = no limit
        self.first_target = str(p("~first_target", "auto")).strip().upper()   # AUTO | A | B
        self.start_tol = float(p("~start_tolerance", 0.10))
        self.on_cancel = str(p("~on_cancel", "stop")).strip().lower()         # stop | resume
        self.cancel_resume_delay = float(p("~cancel_resume_delay", 5.0))
        self.traj_rate = float(p("~traj_rate", 10.0))
        self.summary_period = float(p("~summary_period", 10.0))
        self.bin_size = float(p("~bin_size", 0.2))
        self.stall_window = float(p("~stall_window", 2.0))     # s, 0 disables stall detection
        self.stall_min_dist = float(p("~stall_min_dist", 0.03))
        self.odom_step = float(p("~odom_step", 0.02))          # m, odometer hysteresis vs. pose jitter
        self.map_frame = p("~map_frame", "map")
        self.base_frame = p("~base_frame", "base_link")
        self.tf_timeout = float(p("~tf_timeout", 0.1))
        self.srv_set_name = p("~set_base_x_service", "/ranger/set_base_x")
        self.srv_clear_name = p("~clear_goal_service", "/clear_ranger_goal")
        self.status_topic = p("~status_topic", "/ranger/base_goal_status")

        # go_to_base_x settings, so the logged human rule is the one it applies.
        gtb = str(p("~go_to_base_x_ns", "/go_to_base_x")).rstrip("/")

        def g(name, default):
            return p("~" + name, p(gtb + "/" + name, default))

        self.speed = abs(float(g("speed", 0.1))) or 0.1
        self.h_enable = bool(p("~human_log", True))
        self.h_topic = g("human_topic", "/human_detection_3d")
        self.h_msg_type = g("human_msg_type", "vision_msgs/Detection3DArray")
        self.h_x_range = float(g("human_x_range", 0.7))
        self.h_y_range = float(g("human_y_range", 0.5))
        self.h_min_score = float(g("human_min_score", 0.5))
        self.h_timeout = float(g("human_timeout", 0.5))
        self.resume_delay = float(g("human_resume_delay", 3.0))

        lt = float(p("~leg_timeout", 0.0))   # s of non-human-paused time per leg; 0 = auto
        self.leg_timeout = lt if lt > 0 else 3.0 * self.span / self.speed + 30.0
        self.finish_leg_on_limit = bool(p("~finish_leg_on_limit", True))

        # ---------------- run folder ----------------
        tag = str(p("~run_tag", "")).strip().replace(" ", "_").replace("/", "_")
        root = os.path.expanduser(str(p("~log_dir", "~/ranger_shuttle_logs")))
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_" + tag if tag else "")
        self.run_dir = os.path.join(root, name)
        n = 1
        while os.path.exists(self.run_dir):
            n += 1
            self.run_dir = os.path.join(root, "%s_%d" % (name, n))
        os.makedirs(self.run_dir)

        # ---------------- state ----------------
        self.lock = threading.RLock()
        self.t0 = None
        self.t_last = None
        self.status = ""
        self.status_t = 0.0
        self.status_seen = False
        self.phase = "START"            # START -> LEG <-> DWELL -> DONE
        self.next_end = None
        self.next_send_t = 0.0
        self.end_reason = None
        self.stop_after_leg = None
        self.leg = None
        self.leg_no = 0
        self.stop = None
        self.stop_no = 0
        self.stall_quiet_t = -1e9       # a new stall needs a full window after the last one
        self.last_end = None
        self.travels = 0
        self.pose = None                # (t, x, y, yaw)
        self.pose_ok = None
        self.hist = deque()             # (t, x, y)
        self.vx = 0.0
        self.odom = 0.0
        self.odom_anchor = None
        self.humans = []                # [(frame, (x, y, z))]
        self.humans_t = -1e9
        self.time = {"moving": 0.0, "human": 0.0, "stall": 0.0, "idle": 0.0}
        self.human_stops, self.stall_stops, self.legs_done = [], [], []
        self.next_summary = 0.0
        self.t_start_wall = time.time()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.ev = CsvLog(os.path.join(self.run_dir, "events.csv"), EVENT_HDR)
        self.stops_log = CsvLog(os.path.join(self.run_dir, "stops.csv"), STOP_HDR)
        self.legs_log = CsvLog(os.path.join(self.run_dir, "legs.csv"), LEG_HDR)
        self.traj = CsvLog(os.path.join(self.run_dir, "trajectory.csv"), TRAJ_HDR)
        self.write_config()

        self.srv_set = rospy.ServiceProxy(self.srv_set_name, SetBaseX)
        self.srv_clear = rospy.ServiceProxy(self.srv_clear_name, Trigger)
        rospy.Subscriber(self.status_topic, String, self.cb_status, queue_size=20)
        self.setup_humans()

        rospy.loginfo("[shuttle] A=%.3f m <-> B=%.3f m (span %.3f m), dwell %.1f s, logging to %s",
                      self.x["A"], self.x["B"], self.span, self.dwell, self.run_dir)

    # ------------------------------ setup / io ----------------------------------
    def config_dict(self):
        return {
            "x_a": self.x["A"], "x_b": self.x["B"], "span": rnd(self.span, 4),
            "tolerance": self.tolerance, "dwell": self.dwell,
            "max_travels": self.max_travels, "max_duration": self.max_duration,
            "first_target": self.first_target, "start_tolerance": self.start_tol,
            "on_cancel": self.on_cancel, "cancel_resume_delay": self.cancel_resume_delay,
            "leg_timeout": self.leg_timeout, "finish_leg_on_limit": self.finish_leg_on_limit,
            "traj_rate": self.traj_rate, "summary_period": self.summary_period,
            "bin_size": self.bin_size, "stall_window": self.stall_window,
            "stall_min_dist": self.stall_min_dist, "odom_step": self.odom_step,
            "map_frame": self.map_frame, "base_frame": self.base_frame,
            "set_base_x_service": self.srv_set_name, "clear_goal_service": self.srv_clear_name,
            "status_topic": self.status_topic,
            "go_to_base_x": {
                "speed": self.speed, "human_topic": self.h_topic,
                "human_msg_type": self.h_msg_type, "human_x_range": self.h_x_range,
                "human_y_range": self.h_y_range, "human_min_score": self.h_min_score,
                "human_timeout": self.h_timeout, "human_resume_delay": self.resume_delay,
            },
            "human_log": self.h_enable,
        }

    def write_config(self):
        cfg = {"run_dir": self.run_dir, "start_wall": wall_iso(self.t_start_wall),
               "host": socket.gethostname(),
               "ros_master_uri": os.environ.get("ROS_MASTER_URI", ""),
               "params": self.config_dict()}
        self.write_json("config.json", cfg)

    def write_json(self, name, obj):
        path = os.path.join(self.run_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    def setup_humans(self):
        if not self.h_enable:
            return
        try:                                    # load by name -> no hard vision_msgs dependency
            import roslib.message
            cls = roslib.message.get_message_class(self.h_msg_type)
        except Exception:
            cls = None
        if cls is None:
            rospy.logwarn("[shuttle] human logging OFF: message type '%s' not found.", self.h_msg_type)
            self.h_enable = False
            return
        rospy.Subscriber(self.h_topic, cls, self.cb_humans, queue_size=1)

    def event(self, now, name, detail=""):
        leg = self.leg
        x = y = None
        if self.pose is not None:
            x, y = self.pose[1], self.pose[2]
        self.ev.row([now, self.rel(now), wall_iso(now), name,
                     leg["no"] if leg else None, leg["target_x"] if leg else None,
                     x, y, detail])

    def rel(self, now):
        return None if self.t0 is None else now - self.t0

    # ------------------------------- callbacks ----------------------------------
    def cb_humans(self, msg):
        pts = []
        for det in msg.detections:
            if detection_score(det.results) < self.h_min_score:
                continue
            frame = getattr(getattr(det, "header", None), "frame_id", "") or msg.header.frame_id
            c = det.bbox.center.position
            pts.append((frame.lstrip("/"), (c.x, c.y, c.z)))
        now = rospy.get_time()
        with self.lock:
            self.humans = pts
            self.humans_t = now

    def cb_status(self, msg):
        s = msg.data
        now = rospy.get_time()
        with self.lock:
            if s != self.status:
                self.status_t = now
            self.status = s
            self.status_seen = True
            leg = self.leg
            if leg is None or self.phase == "DONE":
                return
            if s == "MOVING":
                leg["seen_moving"] = True
                if self.stop is not None and self.stop["cause"] == "human":
                    self.close_stop(now, "RESUMED")
            elif s == "PAUSED_HUMAN":
                if not leg["seen_moving"]:
                    return
                if self.stop is not None and self.stop["cause"] == "stall":
                    self.close_stop(now, "HUMAN_PAUSE")
                if self.stop is None:
                    self.open_stop("human", now, now)
            elif s in ("REACHED", "CANCELLED", "IDLE"):
                if not leg["seen_moving"]:
                    return                      # latched status of an earlier goal
                self.finish_leg(now, {"REACHED": "REACHED", "CANCELLED": "CANCELLED",
                                      "IDLE": "LOST"}[s])

    # ------------------------------ geometry ------------------------------------
    def lookup_pose(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rospy.Time(0),
                                                 rospy.Duration(self.tf_timeout))
        except Exception:
            return None
        t = tr.transform.translation
        return t.x, t.y, yaw_from_quat(tr.transform.rotation)

    def body_dir(self):
        """go_to_base_x's travel direction along base_link x (+1/-1), None if no goal."""
        if self.leg is None or self.pose is None:
            return None
        err = self.leg["target_x"] - self.pose[1]
        return math.copysign(1.0, err) * math.copysign(1.0, math.cos(self.pose[3]))

    def human_info(self, now):
        """People seen now, in base_link: count, nearest, nearest one ahead in the lane."""
        out = {"n": 0, "nearest": None, "ahead": None, "lat": None, "block": None,
               "map_x": None, "map_y": None}
        if not self.h_enable or now - self.humans_t > self.h_timeout or not self.humans:
            return out
        cache, pts = {}, []
        for frame, p in self.humans:
            if frame not in cache:
                try:
                    cache[frame] = (
                        self.tf_buffer.lookup_transform(self.base_frame, frame, rospy.Time(0),
                                                        rospy.Duration(0.02)),
                        self.tf_buffer.lookup_transform(self.map_frame, frame, rospy.Time(0),
                                                        rospy.Duration(0.02)))
                except Exception:
                    cache[frame] = None
            if cache[frame] is None:
                continue
            bx, by, _ = transform_point(cache[frame][0], p)
            mx, my, _ = transform_point(cache[frame][1], p)
            pts.append((bx, by, mx, my))
        if not pts:
            return out
        out["n"] = len(pts)
        out["nearest"] = min(math.hypot(b[0], b[1]) for b in pts)
        d = self.body_dir()
        if d is not None:
            lane = [(b[0] * d, b) for b in pts if b[0] * d > 0.0 and abs(b[1]) <= self.h_y_range]
            if lane:
                ahead, b = min(lane, key=lambda e: e[0])
                out.update(ahead=ahead, lat=b[1], block=ahead <= self.h_x_range,
                           map_x=b[2], map_y=b[3])
        return out

    # ------------------------------ bookkeeping ---------------------------------
    def end_of(self, x):
        for e in ("A", "B"):
            if abs(x - self.x[e]) <= self.start_tol:
                return e
        return None

    def open_leg(self, end, now):
        self.leg_no += 1
        x0 = self.pose[1]
        tgt = self.x[end]
        self.leg = {
            "no": self.leg_no, "to_end": end, "from_end": self.end_of(x0),
            "from_x": x0, "target_x": tgt, "dir": 1 if tgt >= x0 else -1,
            "t_start": now, "wall_start": wall_iso(now), "seen_moving": False,
            "odom0": self.odom, "n_human": 0, "human_time": 0.0,
            "n_stall": 0, "stall_time": 0.0, "moving_time": 0.0,
        }
        self.phase = "LEG"
        self.event(now, "LEG_START", "%s -> %s (%.3f -> %.3f m)" % (
            self.leg["from_end"] or "mid", end, x0, tgt))

    def finish_leg(self, now, result):
        leg = self.leg
        if self.stop is not None:
            self.close_stop(now, result)
        x_end = self.pose[1] if self.pose else None
        completes, travel_no = False, None
        if result == "REACHED":
            end = leg["to_end"]
            if self.last_end is not None and self.last_end != end:
                self.travels += 1
                completes, travel_no = True, self.travels
            self.last_end = end
        leg.update(t_end=now, duration=now - leg["t_start"], result=result,
                   completes=completes, travel_no=travel_no, x_end=x_end,
                   final_error=None if x_end is None else leg["target_x"] - x_end,
                   path_len=self.odom - leg["odom0"])
        self.legs_log.row([leg["no"], leg["from_end"], leg["to_end"], leg["from_x"],
                           leg["target_x"], leg["dir"], leg["t_start"], leg["wall_start"],
                           now, leg["duration"], result, completes, travel_no, x_end,
                           leg["final_error"], leg["path_len"], leg["n_human"], leg["human_time"],
                           leg["n_stall"], leg["stall_time"], leg["moving_time"]])
        self.legs_done.append({k: leg[k] for k in (
            "no", "from_end", "to_end", "dir", "duration", "result", "completes",
            "final_error", "n_human", "human_time", "n_stall", "stall_time", "moving_time")})
        self.event(now, "LEG_END", "%s in %.1f s%s" % (
            result, leg["duration"], " (travel %d)" % travel_no if completes else ""))
        rospy.loginfo("[shuttle] leg %d %s in %.1f s, %d human stop(s)%s", leg["no"], result,
                      leg["duration"], leg["n_human"],
                      " -> travel %d" % travel_no if completes else "")
        self.leg = None
        if self.phase == "DONE":
            return
        other = "B" if leg["to_end"] == "A" else "A"
        if result == "REACHED":
            if self.max_travels > 0 and self.travels >= self.max_travels:
                return self.request_end(now, "max_travels=%d reached" % self.max_travels)
            if self.stop_after_leg:
                return self.request_end(now, self.stop_after_leg)
            self.schedule(now, other, self.dwell)
        elif result == "CANCELLED":
            if self.on_cancel == "resume":
                self.event(now, "CANCEL_RESUME", "re-send in %.1f s" % self.cancel_resume_delay)
                self.schedule(now, leg["to_end"], self.cancel_resume_delay)
            else:
                self.request_end(now, "goal cancelled externally (%s)" % self.srv_clear_name)
        elif result == "LOST":
            rospy.logwarn("[shuttle] go_to_base_x reported IDLE mid-leg (restarted?); re-sending.")
            self.schedule(now, leg["to_end"], 2.0)

    def schedule(self, now, end, delay):
        self.phase = "DWELL"
        self.next_end = end
        self.next_send_t = now + delay

    def open_stop(self, cause, t_start, now, start_pose=None):
        leg = self.leg
        sp = start_pose or self.pose or (now, None, None, None)
        h = self.human_info(now) if cause == "human" else {}
        x0 = sp[1]
        self.stop_no += 1
        self.stop = {
            "id": self.stop_no, "cause": cause, "leg": leg["no"], "dir": leg["dir"],
            "to_end": leg["to_end"],
            "target_x": leg["target_x"], "t_start": t_start, "wall_start": wall_iso(t_start),
            "x_start": x0, "y_start": sp[2], "yaw_start": sp[3], "vx_onset": self.vx,
            "dist_from_leg_start": None if x0 is None else abs(x0 - leg["from_x"]),
            "dist_to_target": None if x0 is None else abs(leg["target_x"] - x0),
            "leg_progress": None if x0 is None else ratio(
                abs(x0 - leg["from_x"]), abs(leg["target_x"] - leg["from_x"])),
            "h": h,
        }
        leg["n_" + cause] += 1
        if cause == "stall":                    # the stall started stall_window ago
            back = max(0.0, now - t_start)
            self.time["moving"] -= back
            self.time["stall"] += back
            leg["moving_time"] -= back
            leg["stall_time"] += back
        detail = cause
        if h.get("ahead") is not None:
            detail += " person %.2f m ahead, %.2f m lateral" % (h["ahead"], h["lat"])
        self.event(now, "STOP_START", detail)
        rospy.loginfo("[shuttle] STOP #%d (%s) at x=%.3f, leg %d%s", self.stop_no, cause,
                      x0 if x0 is not None else float("nan"), leg["no"],
                      "" if h.get("ahead") is None else
                      ", person %.2f m ahead" % h["ahead"])

    def close_stop(self, now, reason):
        s = self.stop
        self.stop = None
        dur = now - s["t_start"]
        if s["cause"] == "human" and reason == "RESUMED":
            blocked = max(0.0, dur - self.resume_delay)
        else:
            blocked = dur
        x_end = self.pose[1] if self.pose else None
        drift = None if (x_end is None or s["x_start"] is None) else x_end - s["x_start"]
        h = s["h"]
        self.stops_log.row([s["id"], s["cause"], s["leg"], s["to_end"], s["dir"], s["target_x"],
                            s["t_start"], self.rel(s["t_start"]), s["wall_start"], now, wall_iso(now),
                            dur, blocked, reason, s["x_start"], s["y_start"], s["yaw_start"],
                            x_end, drift, s["leg_progress"], s["dist_from_leg_start"],
                            s["dist_to_target"], s["vx_onset"], h.get("ahead"), h.get("lat"),
                            h.get("map_x"), h.get("map_y"), h.get("n")])
        rec = {"duration": dur, "blocked": blocked, "x": s["x_start"], "to_end": s["to_end"],
               "reason": reason, "h_ahead": h.get("ahead"), "drift": drift}
        (self.human_stops if s["cause"] == "human" else self.stall_stops).append(rec)
        if s["cause"] == "stall":
            self.stall_quiet_t = now
        self.event(now, "STOP_END", "%s %s after %.2f s" % (s["cause"], reason, dur))
        rospy.loginfo("[shuttle] STOP #%d (%s) ended: %s after %.2f s", s["id"], s["cause"],
                      reason, dur)

    def sample(self, now, pose):
        if self.t0 is None:
            self.t0 = now
            self.next_summary = now + self.summary_period
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 1.0)
        self.t_last = now
        if self.leg is not None:
            if self.stop is not None:
                self.time[self.stop["cause"]] += dt
                self.leg[self.stop["cause"] + "_time"] += dt
            else:
                self.time["moving"] += dt
                self.leg["moving_time"] += dt
        else:
            self.time["idle"] += dt

        if pose is None:
            if self.pose_ok is not False:
                self.pose_ok = False
                self.event(now, "POSE_LOST", "TF %s->%s" % (self.map_frame, self.base_frame))
            return
        if self.pose_ok is False:
            self.event(now, "POSE_OK")
        self.pose_ok = True
        x, y, yaw = pose
        self.pose = (now, x, y, yaw)
        self.hist.append((now, x, y))
        keep = max(self.stall_window, 0.5) + 1.0
        while self.hist and now - self.hist[0][0] > keep:
            self.hist.popleft()
        old = next((h for h in self.hist if now - h[0] <= 0.6), None)
        if old is not None and now - old[0] >= 0.2:
            self.vx = (x - old[1]) / (now - old[0])
        if self.odom_anchor is None:
            self.odom_anchor = (x, y)
        else:
            d = math.hypot(x - self.odom_anchor[0], y - self.odom_anchor[1])
            if d >= self.odom_step:
                self.odom += d
                self.odom_anchor = (x, y)

        h = self.human_info(now)
        leg = self.leg
        self.traj.row([now, self.rel(now), leg["no"] if leg else None,
                       leg["target_x"] if leg else None, leg["dir"] if leg else None,
                       self.status, self.stop["cause"] if self.stop else None,
                       x, y, yaw, self.vx, self.odom,
                       h["n"], h["nearest"], h["ahead"], h["lat"], h["block"]], flush=False)

    def check_leg(self, now):
        leg = self.leg
        if leg is None:
            return
        if self.stall_window > 0 and self.pose_ok:
            if self.stop is None:
                if (self.status == "MOVING" and leg["seen_moving"]
                        and now - max(self.status_t, leg["t_start"], self.stall_quiet_t)
                        >= self.stall_window
                        and self.hist and now - self.hist[0][0] >= self.stall_window):
                    ref = next((h for h in self.hist if now - h[0] <= self.stall_window), None)
                    moved = None if ref is None else math.hypot(self.pose[1] - ref[1],
                                                                self.pose[2] - ref[2])
                    if moved is not None and moved < self.stall_min_dist:
                        start = (ref[0], ref[1], ref[2], self.pose[3])
                        self.open_stop("stall", ref[0], now, start)
            elif self.stop["cause"] == "stall":
                s = self.stop
                if s["x_start"] is not None and math.hypot(
                        self.pose[1] - s["x_start"], self.pose[2] - s["y_start"]) >= self.stall_min_dist:
                    self.close_stop(now, "MOVING")
        if leg["moving_time"] + leg["stall_time"] > self.leg_timeout:
            self.event(now, "LEG_TIMEOUT", "%.0f s without arriving" % self.leg_timeout)
            self.request_end(now, "leg %d timeout (%.0f s not paused for a person)" % (
                leg["no"], self.leg_timeout))

    def request_end(self, now, reason):
        if self.end_reason is None:
            self.end_reason = reason
            self.event(now, "END_REQUEST", reason)
            rospy.loginfo("[shuttle] ending run: %s", reason)
        self.phase = "DONE"

    def check_limits(self, now):
        if self.max_duration > 0 and self.t0 is not None and now - self.t0 >= self.max_duration:
            reason = "max_duration=%.0f s reached" % self.max_duration
            if self.leg is not None and self.finish_leg_on_limit:
                if not self.stop_after_leg:
                    self.stop_after_leg = reason
                    self.event(now, "FINISH_AFTER_LEG", reason)
                    rospy.loginfo("[shuttle] %s; finishing the current leg.", reason)
            else:
                self.request_end(now, reason)

    def plan_first(self, now):
        if self.pose is None:
            rospy.logwarn_throttle(5.0, "[shuttle] waiting for TF %s->%s ...",
                                   self.map_frame, self.base_frame)
            return None
        if not self.status_seen and now - self.t0 < 2.0:
            return None                         # let the latched go_to_base_x status arrive
        x = self.pose[1]
        here = self.end_of(x)
        if self.first_target in ("A", "B"):
            end = self.first_target
            if here == end:
                end = "B" if end == "A" else "A"
        elif here is not None:
            end = "B" if here == "A" else "A"
        else:
            end = "A" if abs(x - self.x["A"]) <= abs(x - self.x["B"]) else "B"
        self.last_end = here
        self.event(now, "RUN_START", "x=%.3f, at end %s, first goal %s" % (x, here or "-", end))
        if here is None:
            rospy.loginfo("[shuttle] x=%.3f is not at an end: approach leg to %s (not a travel).",
                          x, end)
        return end

    # ------------------------------- summary ------------------------------------
    def summary(self, now, final):
        run_s = 0.0 if self.t0 is None else now - self.t0
        hs = list(self.human_stops)
        n_open_h = 1 if (self.stop is not None and self.stop["cause"] == "human") else 0
        n_h = len(hs) + n_open_h
        moving = self.time["moving"]
        dist = self.odom
        lo, hi = min(self.x.values()), max(self.x.values())
        nb = max(1, int(math.ceil((hi - lo) / self.bin_size - 1e-9)))
        bins = [{"x_from": rnd(lo + i * self.bin_size), "x_to": rnd(min(hi, lo + (i + 1) * self.bin_size)),
                 "to_A": 0, "to_B": 0} for i in range(nb)]
        outside = 0
        for s in hs:
            if s["x"] is None or s["x"] < lo or s["x"] > hi:
                outside += 1
                continue
            i = min(nb - 1, int((s["x"] - lo) / self.bin_size))
            bins[i]["to_" + s["to_end"]] += 1
        resumed = [s for s in hs if s["reason"] == "RESUMED"]
        travels = [l for l in self.legs_done if l["completes"]]
        results = {}
        for l in self.legs_done:
            results[l["result"]] = results.get(l["result"], 0) + 1
        legs_with_h = sum(1 for l in self.legs_done if l["n_human"] > 0)
        reached = [l for l in self.legs_done if l["result"] == "REACHED"]
        ss = self.stall_stops

        return {
            "status": "finished" if final else "running",
            "end_reason": self.end_reason if final else None,
            "run_dir": self.run_dir,
            "start_wall": wall_iso(self.t_start_wall),
            "updated_wall": wall_iso(now),
            "run_duration_s": rnd(run_s, 2),
            "endpoints": {"A": self.x["A"], "B": self.x["B"], "span_m": rnd(self.span)},
            "travel": {
                "travels_completed": self.travels,
                "round_trips": self.travels // 2,
                "legs_sent": self.leg_no,
                "legs_by_result": results,
                "distance_m": rnd(dist, 2),
                "nominal_distance_m": rnd(self.travels * self.span, 2),
                "travel_time_s": describe([l["duration"] for l in travels]),
                "travel_time_net_s": describe([l["moving_time"] for l in travels]),
                "arrival_error_abs_m": describe([abs(l["final_error"]) for l in reached
                                                 if l["final_error"] is not None]),
            },
            "time_s": {k: rnd(v, 2) for k, v in self.time.items()},
            "human_interventions": {
                "count": n_h,
                "in_progress": bool(n_open_h),
                "legs_with_intervention": legs_with_h,
                "fraction_of_legs": ratio(legs_with_h, len(self.legs_done)),
                "per_travel": ratio(n_h, self.travels),
                "per_100m": ratio(n_h * 100.0, dist),
                "per_hour_run": ratio(n_h * 3600.0, run_s),
                "per_hour_moving": ratio(n_h * 3600.0, moving),
                "mean_distance_between_m": ratio(dist, n_h, 3),
                "mean_moving_time_between_s": ratio(moving, n_h, 2),
                "by_destination": {"to_A": sum(1 for s in hs if s["to_end"] == "A"),
                                   "to_B": sum(1 for s in hs if s["to_end"] == "B")},
                "end_reasons": {r: sum(1 for s in hs if s["reason"] == r)
                                for r in sorted(set(s["reason"] for s in hs))},
                "duration_s": describe([s["duration"] for s in hs]),
                "blocked_s": describe([s["blocked"] for s in resumed]),
                "person_ahead_at_onset_m": describe([s["h_ahead"] for s in hs]),
                "drift_while_stopped_m": describe([abs(s["drift"]) for s in hs
                                                   if s["drift"] is not None]),
                "resume_delay_s": self.resume_delay,
                "location_hist": {"bin_m": self.bin_size, "outside_range": outside, "bins": bins},
            },
            "stalls": {
                "count": len(ss) + (1 if (self.stop is not None and self.stop["cause"] == "stall") else 0),
                "duration_s": describe([s["duration"] for s in ss]),
                "window_s": self.stall_window, "min_dist_m": self.stall_min_dist,
            },
        }

    def write_summary(self, now, final=False):
        self.write_json("summary.json", self.summary(now, final))
        self.traj.flush()
        self.next_summary = now + self.summary_period

    def status_line(self, now):
        run_s = 0.0 if self.t0 is None else now - self.t0
        n_h = len(self.human_stops) + (1 if self.stop and self.stop["cause"] == "human" else 0)
        n_s = len(self.stall_stops) + (1 if self.stop and self.stop["cause"] == "stall" else 0)
        rospy.loginfo("[shuttle] %s | travels %d (%d round trips) | human stops %d (%.1f s) | "
                      "stalls %d | %.1f m | %s", time.strftime("%H:%M:%S", time.gmtime(run_s)),
                      self.travels, self.travels // 2, n_h, self.time["human"], n_s, self.odom,
                      self.status or "-")

    # -------------------------------- run ---------------------------------------
    def send_goal(self, end):
        tgt = self.x[end]
        msg = ""
        for attempt in range(3):
            try:
                resp = self.srv_set(tgt, self.tolerance)
                if resp.accepted:
                    with self.lock:
                        self.event(rospy.get_time(), "GOAL_SENT", resp.message)
                    return
                msg = resp.message
            except (rospy.ServiceException, rospy.ROSException) as e:
                msg = str(e)
            rospy.logwarn("[shuttle] set_base_x(%.3f) failed (%d/3): %s", tgt, attempt + 1, msg)
            time.sleep(1.0)
        with self.lock:
            now = rospy.get_time()
            if self.leg is not None:
                self.finish_leg(now, "REJECTED")
            self.request_end(now, "set_base_x failed: %s" % msg)

    def wait_services(self):
        while not rospy.is_shutdown():
            try:
                self.srv_set.wait_for_service(timeout=5.0)
                break
            except rospy.ROSException:
                rospy.logwarn("[shuttle] waiting for %s (is line_following.launch running?)",
                              self.srv_set_name)
        try:
            self.srv_clear.wait_for_service(timeout=5.0)
        except rospy.ROSException:
            rospy.logwarn("[shuttle] %s not available; Ctrl-C will not be able to stop the base "
                          "mid-leg.", self.srv_clear_name)

    def run(self):
        try:
            self.wait_services()
            rate = rospy.Rate(self.traj_rate)
            while not rospy.is_shutdown():
                now = rospy.get_time()
                pose = self.lookup_pose()
                send = None
                with self.lock:
                    self.sample(now, pose)
                    if self.phase == "LEG":
                        self.check_leg(now)
                    self.check_limits(now)
                    if self.phase == "DONE":
                        break
                    if self.phase == "START":
                        send = self.plan_first(now)
                    elif self.phase == "DWELL" and now >= self.next_send_t and pose is not None:
                        send = self.next_end
                    if send is not None:
                        self.open_leg(send, now)
                    if now >= self.next_summary:
                        self.write_summary(now)
                        self.status_line(now)
                if send is not None:
                    self.send_goal(send)
                rate.sleep()
        except (KeyboardInterrupt, rospy.ROSInterruptException):
            with self.lock:
                self.request_end(rospy.get_time(), "stopped by operator (Ctrl-C / SIGTERM)")
        self.finalize()

    def finalize(self):
        with self.lock:
            self.phase = "DONE"
            active = self.leg is not None
        if active:
            try:
                self.srv_clear()
                rospy.loginfo("[shuttle] active goal cancelled; base stopped.")
            except Exception as e:
                rospy.logerr("[shuttle] could not cancel the goal (%s): the base will finish "
                             "the current leg on its own.", e)
        with self.lock:
            now = rospy.get_time()
            if self.leg is not None:
                self.finish_leg(now, "ABORTED")
            if self.end_reason is None:
                self.end_reason = "node shutdown"
            self.event(now, "RUN_END", self.end_reason)
            summ = self.summary(now, True)
            self.write_json("summary.json", summ)
            for f in (self.traj, self.ev, self.stops_log, self.legs_log):
                f.close()
        hi = summ["human_interventions"]
        rospy.loginfo("[shuttle] done (%s): %d travels, %.1f m, %d human stops (%s/100 m), "
                      "%d stalls. Logs: %s", self.end_reason, summ["travel"]["travels_completed"],
                      self.odom, hi["count"], hi["per_100m"], summ["stalls"]["count"], self.run_dir)


def _sigterm(_signum, _frame):
    raise KeyboardInterrupt()


def main():
    # Signals are handled here so that Ctrl-C can still call /clear_ranger_goal.
    rospy.init_node("ranger_shuttle", disable_signals=True)
    signal.signal(signal.SIGTERM, _sigterm)
    node = Shuttle()
    node.run()
    rospy.signal_shutdown("shuttle finished")


if __name__ == "__main__":
    main()
