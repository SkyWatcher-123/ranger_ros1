#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Localize the AgileX Ranger Mini base from apriltag_ros /tag_detections and
publish the result as the 3 planar joint states that drive  odom -> base_link
in the URDF (rangerxarm_abb.urdf.xacro).

This node is the AprilTag-localization counterpart of cmd_vel_to_odom.py:

    cmd_vel_to_odom.py           integrates /cmd_vel      -> ranger/joint_states
                                 (drift-free in pure sim, drifts on real robot)
    apriltag_to_ranger_joint_state.py  reads absolute tags -> ranger/joint_states
                                 (drift-free, but only while tags are in view)

Both feed the SAME 3 joints, so run only ONE at a time:
  * cmd_vel integration for pure RViz simulation,
  * this AprilTag node for the real robot.

URDF joint chain (odom -> base_link), see rangerxarm_abb.urdf.xacro:

    odom_to_ranger_base_x   prismatic x   <- published x
    ranger_base_x_to_y      prismatic y   <- published y
    ranger_base_y_to_yaw    revolute  z   <- published yaw

The constant base height (the first joint's origin z = 0.346 m) and the
flat-ground assumption (roll = pitch = 0, base_link z-axis points straight up)
are baked into the URDF joint structure, so this node only has to output
x, y and yaw. We deliberately DO NOT broadcast odom -> base_link here;
robot_state_publisher builds that TF from the joint states we publish.

Pose math (identical to the already-verified apriltag_to_odom.py):

    known:    T_odom_tag   (rosparam tag_locations)
    measured: T_cam_tag    (AprilTagDetection pose; tag expressed in camera frame)
    known:    T_base_cam   (TF tree: base_link -> loc_camera optical frame)

    T_odom_base = T_odom_tag * inv(T_cam_tag) * inv(T_base_cam)

Then  x, y = translation(T_odom_base);  yaw = yaw(T_odom_base).
Multiple tags in one message are distance-filtered and fused (weighted mean
for x/y, weighted circular mean for yaw).
"""

import math
import statistics

import rospy
import tf2_ros
from tf import transformations as tft

from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AprilTagToRangerJointState:
    def __init__(self):
        # --- Frames / topics -------------------------------------------------
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        # Camera frame the tag pose is expressed in.
        # Leave EMPTY (recommended) to use the frame_id carried inside each
        # detection -- that is the camera's *optical* frame (e.g.
        # loc_camera_color_optical_frame) and is guaranteed to match T_cam_tag.
        # Only set this if you know it matches the detection's frame_id exactly;
        # a mismatched frame silently corrupts the estimate.
        self.camera_frame = rospy.get_param("~camera_frame", "")
        self.tag_topic = rospy.get_param("~tag_topic", "/tag_detections")
        self.js_topic = rospy.get_param("~joint_states_topic", "/ranger/joint_states")

        # --- Joint names (MUST match rangerxarm_abb.urdf.xacro) --------------
        self.joint_x = rospy.get_param("~joint_x", "odom_to_ranger_base_x")
        self.joint_y = rospy.get_param("~joint_y", "ranger_base_x_to_y")
        self.joint_yaw = rospy.get_param("~joint_yaw", "ranger_base_y_to_yaw")

        # --- Fusion / filtering (same meaning as apriltag_to_odom.py) --------
        self.max_tag_distance = float(rospy.get_param("~max_tag_distance", 2.0))
        # weight = 1/dist^weight_power; 0.0 => all tags equal
        self.weight_power = float(rospy.get_param("~weight_power", 0.0))
        # exponential smoothing on the output; 1.0 => no smoothing
        self.alpha = float(rospy.get_param("~smoothing_alpha", 1.0))

        # --- Loop timing -----------------------------------------------------
        self.rate = float(rospy.get_param("~rate", 3.0))
        self.stale_timeout = float(rospy.get_param("~stale_timeout", 1.0))

        # --- Optional nav_msgs/Odometry (joint states are the real product) --
        self.publish_odom = bool(rospy.get_param("~publish_odom", False))
        # Only used for the optional odom z; base height is structural in the URDF.
        self.base_height = float(rospy.get_param("~base_height", 0.346))

        # --- Initial pose, published until the first tag fix -----------------
        # Keeps the odom->base_link TF alive from startup. On the first valid
        # detection this is overwritten by the measured pose.
        self.x = float(rospy.get_param("~init_x", 0.0))
        self.y = float(rospy.get_param("~init_y", 0.0))
        self.yaw = float(rospy.get_param("~init_yaw", 0.0))

        # --- Known tag poses: list of dicts under ~tag_locations or /tag_locations
        tag_list = rospy.get_param("~tag_locations", None)
        if tag_list is None:
            tag_list = rospy.get_param("/tag_locations", [])

        self.T_odom_tag = {}
        for item in tag_list:
            tid = int(item["id"])
            self.T_odom_tag[tid] = tft.concatenate_matrices(
                tft.translation_matrix([float(item["x"]), float(item["y"]),
                                        float(item.get("z", 0.0))]),
                tft.quaternion_matrix([float(item.get("qx", 0.0)), float(item.get("qy", 0.0)),
                                       float(item.get("qz", 0.0)), float(item.get("qw", 1.0))]),
            )
        if not self.T_odom_tag:
            rospy.logwarn("No tag_locations loaded. Load apriltag_localization*.yaml as rosparam.")

        # --- TF input only (base_link -> camera). We never broadcast odom->base.
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.have_fix = False
        self.last_update = rospy.Time(0)

        # --- Settle-and-latch (freeze base pose after arrival, so the arm sees a
        #     static TF instead of AprilTag placement jitter). Triggered by the
        #     ~hold_signal (Bool) topic -- wire it to go_to_base_x's at_target. ---
        self.enable_hold = bool(rospy.get_param("~enable_hold", True))
        self.settle_time = float(rospy.get_param("~settle_time", 1.5))       # s to average after arrival
        self.min_hold_samples = int(rospy.get_param("~min_hold_samples", 5))
        self.hold_max_std = float(rospy.get_param("~hold_max_std", 0.05))    # m; refuse to latch if noisier
        self.hold_state = "LIVE"          # LIVE -> SETTLING -> HELD
        self.hold_signal = False
        self.settle_start = rospy.Time(0)
        self.settle_samples = []
        self.frozen = None                # (x, y, yaw) held pose

        self.pub_js = rospy.Publisher(self.js_topic, JointState, queue_size=10)
        self.pub_odom = rospy.Publisher("odom", Odometry, queue_size=10) if self.publish_odom else None
        # Handshake for the arm: True only once the base pose is frozen and arm-safe.
        self.pub_locked = rospy.Publisher("~pose_locked", Bool, queue_size=1, latch=True)
        self.pub_locked.publish(Bool(data=False))

        rospy.Subscriber(self.tag_topic, AprilTagDetectionArray, self.cb_tags, queue_size=10)
        rospy.Subscriber("~hold_signal", Bool, self.cb_hold, queue_size=1)

        rospy.loginfo(
            "[apriltag_to_ranger_joint_state] out=%s joints=[%s, %s, %s] base=%s "
            "cam=%s rate=%.1fHz tags=%d",
            self.js_topic, self.joint_x, self.joint_y, self.joint_yaw,
            self.base_frame, (self.camera_frame or "<from detection>"),
            self.rate, len(self.T_odom_tag),
        )

    @staticmethod
    def _T_from_tf(tr):
        """geometry_msgs/TransformStamped -> 4x4 homogeneous matrix."""
        t = tr.transform.translation
        q = tr.transform.rotation
        return tft.concatenate_matrices(
            tft.translation_matrix([t.x, t.y, t.z]),
            tft.quaternion_matrix([q.x, q.y, q.z, q.w]),
        )

    def cb_tags(self, msg: AprilTagDetectionArray):
        if not msg.detections:
            return

        candidates = []  # (x, y, yaw, weight)

        for det in msg.detections:
            if not det.id:
                continue
            tid = int(det.id[0])
            if tid not in self.T_odom_tag:
                continue

            # Frame the tag pose is expressed in (optical frame of loc_camera).
            cam_frame = self.camera_frame or det.pose.header.frame_id or msg.header.frame_id
            if not cam_frame:
                rospy.logwarn_throttle(2.0, "AprilTag detection has empty frame_id and ~camera_frame unset.")
                continue

            # Tag pose in camera frame.
            p = det.pose.pose.pose.position
            q = det.pose.pose.pose.orientation
            dist = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            if dist > self.max_tag_distance:
                continue

            T_cam_tag = tft.concatenate_matrices(
                tft.translation_matrix([p.x, p.y, p.z]),
                tft.quaternion_matrix([q.x, q.y, q.z, q.w]),
            )

            # base_link -> camera frame from the TF tree (published from the URDF).
            try:
                tr_base_cam = self.tf_buffer.lookup_transform(
                    self.base_frame, cam_frame, rospy.Time(0), rospy.Duration(0.15)
                )
            except Exception as e:
                rospy.logwarn_throttle(2.0, "Missing TF %s->%s: %s" % (self.base_frame, cam_frame, e))
                continue
            T_base_cam = self._T_from_tf(tr_base_cam)

            # Core equation (same as apriltag_to_odom.py):
            #   T_odom_base = T_odom_tag * inv(T_cam_tag) * inv(T_base_cam)
            T_odom_base = tft.concatenate_matrices(
                self.T_odom_tag[tid],
                tft.inverse_matrix(T_cam_tag),
                tft.inverse_matrix(T_base_cam),
            )

            # Flat-ground assumption: take x, y and yaw only; z/roll/pitch are
            # structural in the URDF joint chain, so we drop them here.
            x, y, _z = tft.translation_from_matrix(T_odom_base)
            yaw = tft.euler_from_quaternion(tft.quaternion_from_matrix(T_odom_base))[2]

            w = 1.0 if self.weight_power <= 0.0 else 1.0 / (max(dist, 1e-6) ** self.weight_power)
            candidates.append((float(x), float(y), float(yaw), float(w)))

        if not candidates:
            return

        wsum = sum(c[3] for c in candidates)
        x = sum(c[0] * c[3] for c in candidates) / wsum
        y = sum(c[1] * c[3] for c in candidates) / wsum
        # Weighted circular mean for yaw (robust to +-pi wrap-around).
        s = sum(math.sin(c[2]) * c[3] for c in candidates) / wsum
        cc = sum(math.cos(c[2]) * c[3] for c in candidates) / wsum
        yaw = math.atan2(s, cc)

        # Optional exponential smoothing.
        if not self.have_fix or self.alpha >= 0.999:
            self.x, self.y, self.yaw = x, y, yaw
        else:
            a = max(0.0, min(1.0, self.alpha))
            self.x += a * (x - self.x)
            self.y += a * (y - self.y)
            self.yaw = wrap_to_pi(self.yaw + a * wrap_to_pi(yaw - self.yaw))

        self.have_fix = True
        self.last_update = rospy.Time.now()

    def cb_hold(self, msg: Bool):
        if not self.enable_hold:
            return
        want = bool(msg.data)
        if want and not self.hold_signal:
            # Arrival: begin the settle window (average the pose, then latch).
            self.hold_state = "SETTLING"
            self.settle_start = rospy.Time.now()
            self.settle_samples = []
            self.pub_locked.publish(Bool(data=False))
            rospy.loginfo("[apriltag_to_ranger_joint_state] arrival -> settling %.1fs before latch.",
                          self.settle_time)
        elif not want and self.hold_signal:
            # New move commanded: release the freeze, resume live localization.
            self.hold_state = "LIVE"
            self.frozen = None
            self.pub_locked.publish(Bool(data=False))
            rospy.loginfo("[apriltag_to_ranger_joint_state] hold released -> live localization.")
        self.hold_signal = want

    def _latch_from_samples(self):
        n = len(self.settle_samples)
        if n < self.min_hold_samples:
            rospy.logwarn("[apriltag_to_ranger_joint_state] only %d settle samples; staying LIVE.", n)
            self.hold_state = "LIVE"
            return
        xs = [s[0] for s in self.settle_samples]
        ys = [s[1] for s in self.settle_samples]
        stdx = statistics.pstdev(xs)
        stdy = statistics.pstdev(ys)
        if max(stdx, stdy) > self.hold_max_std:
            rospy.logwarn("[apriltag_to_ranger_joint_state] pose not settled "
                          "(std x=%.1fmm y=%.1fmm > %.0fmm); staying LIVE.",
                          stdx * 1000, stdy * 1000, self.hold_max_std * 1000)
            self.hold_state = "LIVE"
            return
        mx = statistics.median(xs)
        my = statistics.median(ys)
        syaw = sum(math.sin(s[2]) for s in self.settle_samples) / n
        cyaw = sum(math.cos(s[2]) for s in self.settle_samples) / n
        myaw = math.atan2(syaw, cyaw)
        self.frozen = (mx, my, myaw)
        self.hold_state = "HELD"
        self.pub_locked.publish(Bool(data=True))
        rospy.loginfo("[apriltag_to_ranger_joint_state] LATCHED x=%.3f y=%.3f yaw=%.2fdeg "
                      "(%d samples, std x=%.1fmm y=%.1fmm) -> arm-safe.",
                      mx, my, math.degrees(myaw), n, stdx * 1000, stdy * 1000)

    def _pose_to_publish(self, now):
        """State machine -> (x, y, yaw) to output this tick."""
        if self.hold_state == "SETTLING":
            if self.have_fix:
                self.settle_samples.append((self.x, self.y, self.yaw))
            if (now - self.settle_start).to_sec() >= self.settle_time:
                self._latch_from_samples()
            return self.x, self.y, self.yaw          # keep publishing live while settling
        if self.hold_state == "HELD" and self.frozen is not None:
            return self.frozen                        # frozen -> zero jitter for the arm
        return self.x, self.y, self.yaw               # LIVE

    def publish(self, now, px, py, pyaw):
        yaw = wrap_to_pi(pyaw)  # revolute joint limit is +-pi in the URDF

        js = JointState()
        js.header.stamp = now
        js.name = [self.joint_x, self.joint_y, self.joint_yaw]
        js.position = [px, py, yaw]
        # velocity/effort intentionally left empty; robot_state_publisher only needs position.
        self.pub_js.publish(js)

        if self.pub_odom is not None:
            qx, qy, qz, qw = tft.quaternion_from_euler(0.0, 0.0, yaw)
            od = Odometry()
            od.header.stamp = now
            od.header.frame_id = self.odom_frame
            od.child_frame_id = self.base_frame
            od.pose.pose.position.x = px
            od.pose.pose.position.y = py
            od.pose.pose.position.z = self.base_height
            od.pose.pose.orientation.x = qx
            od.pose.pose.orientation.y = qy
            od.pose.pose.orientation.z = qz
            od.pose.pose.orientation.w = qw
            self.pub_odom.publish(od)

    def spin(self):
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            px, py, pyaw = self._pose_to_publish(now)
            if not self.have_fix:
                rospy.logwarn_throttle(
                    5.0, "No AprilTag fix yet; publishing init pose (x=%.2f y=%.2f yaw=%.2f)."
                    % (self.x, self.y, self.yaw))
            elif self.hold_state == "LIVE":
                age = (now - self.last_update).to_sec()
                if age > self.stale_timeout:
                    rospy.logwarn_throttle(
                        2.0, "AprilTag localization stale (%.2fs); holding last joint states." % age)
            self.publish(now, px, py, pyaw)
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("apriltag_to_ranger_joint_state")
    AprilTagToRangerJointState().spin()
