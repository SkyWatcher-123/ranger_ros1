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

Note: this node owns ~cmd_vel_out. It publishes zero to actively hold a stop
after a goal completes; before the first goal it stays silent so you can drive
manually into the same topic for bring-up.
"""

import math
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64

from ranger_msgs.srv import SetBaseX, SetBaseXResponse


def yaw_from_quat(q):
    # yaw about z
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


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

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_cmd = rospy.Publisher("~cmd_vel_out", Twist, queue_size=10)
        self.pub_done = rospy.Publisher("~at_target", Bool, queue_size=1, latch=True)
        self.pub_err = rospy.Publisher("~x_error", Float64, queue_size=10)

        self.srv = rospy.Service("~set_base_x", SetBaseX, self.cb_set_base_x)
        rospy.loginfo("[go_to_base_x] ready. speed=%.3f m/s default_tol=%.3f m frames %s->%s",
                      self.speed, self.default_tol, self.map_frame, self.base_frame)

    def cb_set_base_x(self, req):
        self.target_x = float(req.target_x)
        self.tolerance = float(req.tolerance) if req.tolerance and req.tolerance > 0.0 else self.default_tol
        self.active = True
        self.started = True
        self.pub_done.publish(Bool(data=False))
        msg = "Goal accepted: target_x=%.3f m (tol=%.3f m)" % (self.target_x, self.tolerance)
        rospy.loginfo("[go_to_base_x] %s", msg)
        return SetBaseXResponse(accepted=True, message=msg)

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
                        self.pub_cmd.publish(Twist())          # stop
                        self.pub_done.publish(Bool(data=True))
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
                            cmd = Twist()
                            cmd.linear.x = vx
                            self.pub_cmd.publish(cmd)
                elif self.started:
                    self.pub_cmd.publish(Twist())              # hold stop between goals
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("go_to_base_x")
    GoToBaseX().spin()
