#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roll the Ranger base to a target X coordinate in the map frame, along the tape.

Offers the ranger_msgs/SetBaseX service (target_x, tolerance). While a goal is
active it commands a strictly uniform +/-0.1 m/s on ~cmd_vel_out (which feeds
line_keeper, so the tape controller keeps it straight); it stops the instant
|target_x - current_x| <= tolerance and latches ~at_target = True.

Current X (and heading, to pick the travel direction) is read from TF
map -> base_link, i.e. the fused AprilTag pose via robot_state_publisher.

Human pause (optional; inert when no detections arrive):
  A person from ~human_topic that is AHEAD in the travel direction, within
  ~human_x_range [m] and |y| <= ~human_y_range [m] of ~human_ref_frame
  (default: ~base_frame; no bumper offset), makes the base hold still. The goal stays ACTIVE (not
  marked done). Motion resumes once the path has been clear continuously for
  ~human_resume_delay seconds. Detector not running / stale detections -> old
  behaviour.

Manual cancel: ~clear_ranger_goal (std_srvs/Trigger) drops the active goal and
stops the base (at_target stays False). Send a new set_base_x goal to go again.

Status: ~goal_status (std_msgs/String, latched):
  IDLE | MOVING | PAUSED_HUMAN | REACHED | CANCELLED

Note: this node owns ~cmd_vel_out. It publishes zero to actively hold a stop
after a goal completes; before the first goal it stays silent so you can drive
manually into the same topic for bring-up.
"""

import math
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import Trigger, TriggerResponse

from ranger_msgs.srv import SetBaseX, SetBaseXResponse


# ----------------------------- small pure helpers ------------------------------
def yaw_from_quat(q):
    # yaw about z
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def transform_point(tr, p):
    """Apply a geometry_msgs/TransformStamped to a point p=(x, y, z)."""
    q, t = tr.transform.rotation, tr.transform.translation
    x, y, z = p
    # v' = v + 2w(u x v) + 2 u x (u x v), with u = (qx, qy, qz)
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


def path_blocker(points_xy, direction, x_range, y_range):
    """Closest person (x, y) [base_link] AHEAD in the travel direction with
    0 < x*direction <= x_range and |y| <= y_range; None if the path is clear."""
    best, best_ahead = None, None
    if direction == 0:
        return None
    for x, y in points_xy:
        ahead = x * direction
        if 0.0 < ahead <= x_range and abs(y) <= y_range:
            if best_ahead is None or ahead < best_ahead:
                best, best_ahead = (x, y), ahead
    return best


class PauseLatch(object):
    """Paused while blocked; released only after `delay` s of continuously clear path."""

    def __init__(self, delay):
        self.delay = delay
        self.last_blocked = None          # time [s] the path was last seen blocked

    def reset(self):
        self.last_blocked = None

    def update(self, blocked, now):
        if blocked:
            self.last_blocked = now
        return self.last_blocked is not None and (now - self.last_blocked) < self.delay

    def remaining(self, now):
        return 0.0 if self.last_blocked is None else max(0.0, self.delay - (now - self.last_blocked))


# ---------------------------------- the node -----------------------------------
class GoToBaseX:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.speed = abs(float(rospy.get_param("~speed", 0.1)))        # strictly uniform travel speed
        self.default_tol = float(rospy.get_param("~default_tolerance", 0.03))   # 3 cm (2-5 cm range)
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        # If the base heading is nearly perpendicular to map-x we can't resolve the
        # travel direction safely -> refuse to drive.
        self.min_cos = float(rospy.get_param("~min_cos_heading", 0.3))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.3))

        self.target_x = 0.0
        self.tolerance = self.default_tol
        self.active = False
        self.started = False
        self.status = ""

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_cmd = rospy.Publisher("~cmd_vel_out", Twist, queue_size=10)
        self.pub_done = rospy.Publisher("~at_target", Bool, queue_size=1, latch=True)
        self.pub_err = rospy.Publisher("~x_error", Float64, queue_size=10)
        self.pub_status = rospy.Publisher("~goal_status", String, queue_size=1, latch=True)
        self.set_status("IDLE")

        self._setup_human_pause()

        self.srv = rospy.Service("~set_base_x", SetBaseX, self.cb_set_base_x)
        self.srv_clear = rospy.Service("~clear_ranger_goal", Trigger, self.cb_clear_goal)
        rospy.loginfo("[go_to_base_x] ready. speed=%.3f m/s default_tol=%.3f m frames %s->%s",
                      self.speed, self.default_tol, self.map_frame, self.base_frame)

    # ------------------------------ human pause ---------------------------------
    def _setup_human_pause(self):
        self.human_enable = bool(rospy.get_param("~human_pause_enable", True))
        self.human_x_range = float(rospy.get_param("~human_x_range", 0.7))
        self.human_y_range = float(rospy.get_param("~human_y_range", 0.5))
        self.human_min_score = float(rospy.get_param("~human_min_score", 0.5))
        self.human_timeout = float(rospy.get_param("~human_timeout", 0.5))   # older detections = none
        # Frame the distances are measured in (its +x must be the robot's forward axis).
        self.human_frame = rospy.get_param("~human_ref_frame", "") or self.base_frame
        self.pause = PauseLatch(float(rospy.get_param("~human_resume_delay", 3.0)))
        self.humans = []                  # [(x, y)] in base_link from the latest message
        self.humans_t = rospy.Time(0)     # receive time of the latest message
        if not self.human_enable:
            rospy.loginfo("[go_to_base_x] human pause disabled.")
            return
        topic = rospy.get_param("~human_topic", "/human_detection_3d")
        msg_type = rospy.get_param("~human_msg_type", "vision_msgs/Detection3DArray")
        try:                              # load the message type by name -> no hard dependency
            import roslib.message
            cls = roslib.message.get_message_class(msg_type)
        except Exception:
            cls = None
        if cls is None:
            rospy.logwarn("[go_to_base_x] human pause OFF: message type '%s' not found.", msg_type)
            self.human_enable = False
            return
        rospy.Subscriber(topic, cls, self.cb_humans, queue_size=1)
        rospy.loginfo("[go_to_base_x] human pause ON: %s ahead<=%.2f m |y|<=%.2f m of %s, resume after %.1f s",
                      topic, self.human_x_range, self.human_y_range, self.human_frame, self.pause.delay)

    def cb_humans(self, msg):
        pts, tf_cache = [], {}
        for det in msg.detections:
            if detection_score(det.results) < self.human_min_score:
                continue
            frame = getattr(getattr(det, "header", None), "frame_id", "") or msg.header.frame_id
            if frame not in tf_cache:
                try:
                    tf_cache[frame] = self.tf_buffer.lookup_transform(
                        self.human_frame, frame, rospy.Time(0), rospy.Duration(0.05))
                except Exception as e:
                    tf_cache[frame] = None
                    rospy.logwarn_throttle(5.0, "[go_to_base_x] human TF %s->%s: %s",
                                           frame, self.human_frame, e)
            if tf_cache[frame] is None:
                continue
            c = det.bbox.center.position
            x, y, _ = transform_point(tf_cache[frame], (c.x, c.y, c.z))
            pts.append((x, y))
        self.humans = pts
        self.humans_t = rospy.Time.now()

    def human_blocker(self, vx):
        """Person blocking the path for travel direction sign(vx), else None."""
        if not self.human_enable:
            return None
        if (rospy.Time.now() - self.humans_t).to_sec() > self.human_timeout:
            return None                   # detector absent / stale -> old workflow
        return path_blocker(self.humans, math.copysign(1.0, vx),
                            self.human_x_range, self.human_y_range)

    # ------------------------------- services -----------------------------------
    def cb_set_base_x(self, req):
        self.target_x = float(req.target_x)
        self.tolerance = float(req.tolerance) if req.tolerance and req.tolerance > 0.0 else self.default_tol
        self.active = True
        self.started = True
        self.pause.reset()
        self.pub_done.publish(Bool(data=False))
        self.set_status("MOVING")
        msg = "Goal accepted: target_x=%.3f m (tol=%.3f m)" % (self.target_x, self.tolerance)
        rospy.loginfo("[go_to_base_x] %s", msg)
        return SetBaseXResponse(accepted=True, message=msg)

    def cb_clear_goal(self, _req):
        if not self.active:
            return TriggerResponse(success=True, message="no active goal")
        self.active = False
        self.pause.reset()
        self.pub_cmd.publish(Twist())     # stop now; spin() keeps holding zero
        self.set_status("CANCELLED")
        msg = "goal target_x=%.3f cancelled; base stopped (not marked done)" % self.target_x
        rospy.logwarn("[go_to_base_x] %s", msg)
        return TriggerResponse(success=True, message=msg)

    def set_status(self, status, blocker=None):
        if status == self.status:
            return
        extra = "" if blocker is None else " (person at x=%.2f y=%.2f m in %s)" % (
            blocker[0], blocker[1], getattr(self, "human_frame", self.base_frame))
        rospy.loginfo("[go_to_base_x] status %s -> %s%s", self.status or "-", status, extra)
        self.status = status
        self.pub_status.publish(String(data=status))

    # -------------------------------- main loop ---------------------------------
    def current_pose(self):
        try:
            tr = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(self.tf_timeout))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[go_to_base_x] TF %s->%s: %s",
                                   self.map_frame, self.base_frame, e)
            return None
        return tr.transform.translation.x, yaw_from_quat(tr.transform.rotation)

    def spin(self):
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            pose = self.current_pose()
            if pose is not None:
                cur_x, yaw = pose
                err = self.target_x - cur_x
                self.pub_err.publish(Float64(data=err))

                if self.active:
                    if abs(err) <= self.tolerance:
                        self.active = False
                        self.pause.reset()
                        self.pub_cmd.publish(Twist())          # stop
                        self.pub_done.publish(Bool(data=True))
                        self.set_status("REACHED")
                        rospy.loginfo("[go_to_base_x] reached target_x=%.3f (err=%.3f m). Stopped.",
                                      self.target_x, err)
                    else:
                        cos_y = math.cos(yaw)
                        if abs(cos_y) < self.min_cos:
                            rospy.logwarn_throttle(
                                1.0, "[go_to_base_x] base heading not aligned with map-x "
                                     "(cos=%.2f); refusing to drive.", cos_y)
                            self.pub_cmd.publish(Twist())
                        else:
                            # body +x maps to map-x by cos(yaw); pick body vx so map-x -> target
                            vx = self.speed * math.copysign(1.0, err) * math.copysign(1.0, cos_y)
                            blocker = self.human_blocker(vx)
                            now = rospy.Time.now().to_sec()
                            if self.pause.update(blocker is not None, now):
                                self.pub_cmd.publish(Twist())  # hold still; goal stays ACTIVE
                                self.set_status("PAUSED_HUMAN", blocker)
                                if blocker is None:
                                    rospy.loginfo_throttle(
                                        1.0, "[go_to_base_x] path clear - resuming in %.1f s"
                                        % self.pause.remaining(now))
                            else:
                                cmd = Twist()
                                cmd.linear.x = vx
                                self.pub_cmd.publish(cmd)
                                self.set_status("MOVING")
                elif self.started:
                    self.pub_cmd.publish(Twist())              # hold stop between goals
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("go_to_base_x")
    GoToBaseX().spin()
