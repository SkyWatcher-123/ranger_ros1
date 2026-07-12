#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yellow-tape line-keeper for the Ranger Mini (Stanley-flavored cmd_vel filter).

The Ranger switches motion mode from the Twist it receives:
  * linear.y != 0        -> parallel/side-slip (heading unchanged)
  * linear.y == 0, wz    -> dual-Ackermann (wz -> steering angle) / spin if tight
So we keep linear.y == 0 (stay in Ackermann, no wheel-mode thrash) and correct
heading + lateral drift purely with angular.z:

    wz = k_heading * heading_err  +  k_cross * cross_track_err * sign(v)

  * two terms = heading + cross-track (Stanley reduces to this at low speed);
  * the sign(v) on the cross-track term is essential -- it flips when reversing,
    otherwise backing up diverges like a car steered in reverse;
  * heading term is NOT flipped.

It passes the travel speed (linear.x) straight through from ~cmd_vel_in (that is
where go_to_base_x commands a strict +/-0.1 m/s), clamps it to ~max_speed for
safety, and forces linear.y = 0. Errors come from tape_line_detector on ~tape
(geometry_msgs/Vector3Stamped: x=cross_track [m], y=heading [rad], z=confidence).

Safety: if the tape is stale or low-confidence, it STOPS travel (v=0, wz=0).
"""

import math
import rospy
from geometry_msgs.msg import Twist, Vector3Stamped


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class LineKeeper:
    def __init__(self):
        # Control gains
        self.k_heading = float(rospy.get_param("~k_heading", 1.2))     # wz per rad
        self.k_cross = float(rospy.get_param("~k_cross", 1.5))         # wz per m (times sign(v))
        self.sign = float(rospy.get_param("~sign", 1.0))              # global flip if your y-convention differs

        # Limits / shaping
        self.max_wz = float(rospy.get_param("~max_wz", 0.4))          # rad/s
        self.wz_slew = float(rospy.get_param("~wz_slew", 1.5))        # rad/s^2
        self.max_speed = float(rospy.get_param("~max_speed", 0.1))    # m/s clamp on linear.x
        self.v_min = float(rospy.get_param("~v_min", 0.02))          # below this, don't steer (avoid spin mode)
        # Lateral tolerance band: within +/-deadband_ct [m] no cross-track correction
        # is applied (heading is still held). This is the knob to trade tightness vs.
        # calm steering -- set it to your acceptable lateral drift (e.g. 0.02-0.05),
        # or ~0 and raise the gains to probe the tightest accuracy the loop can hold.
        self.deadband_ct = float(rospy.get_param("~deadband_ct", 0.005))   # m
        self.deadband_hd = float(rospy.get_param("~deadband_hd", 0.0087))  # rad (~0.5 deg)

        # Freshness / safety
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.tape_timeout = float(rospy.get_param("~tape_timeout", 0.4))    # s
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.5))      # s
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.3))
        self.stop_on_tape_loss = bool(rospy.get_param("~stop_on_tape_loss", True))

        # Live tuning: re-read the gain/tolerance params periodically so you can
        # `rosparam set /line_keeper/deadband_ct 0.03` (etc.) mid-run without relaunch.
        self.live_reconfig = bool(rospy.get_param("~live_reconfig", True))
        self.reconfig_period = float(rospy.get_param("~reconfig_period", 0.5))  # s
        self._tunables = ["k_heading", "k_cross", "sign", "max_wz",
                          "wz_slew", "deadband_ct", "deadband_hd"]
        self._last_reconfig = rospy.Time(0)

        # Lateral-accuracy meter: RMS / max |cross_track| while actually rolling,
        # logged every stats_period so you can quantify how tight it actually holds.
        self.stats_period = float(rospy.get_param("~stats_period", 2.0))  # s; <=0 disables
        self._reset_stats()
        self._last_stats = rospy.Time(0)

        self.tape = None
        self.tape_t = rospy.Time(0)
        self.cmd_vx = 0.0
        self.cmd_t = rospy.Time(0)
        self.wz = 0.0

        self.pub = rospy.Publisher("~cmd_vel_out", Twist, queue_size=10)
        rospy.Subscriber("~cmd_vel_in", Twist, self.cb_cmd, queue_size=10)
        rospy.Subscriber("~tape", Vector3Stamped, self.cb_tape, queue_size=10)

        rospy.loginfo("[line_keeper] k_h=%.2f k_ct=%.2f max_wz=%.2f max_speed=%.2f stop_on_loss=%s",
                      self.k_heading, self.k_cross, self.max_wz, self.max_speed, self.stop_on_tape_loss)

    def cb_cmd(self, msg: Twist):
        self.cmd_vx = clamp(msg.linear.x, -self.max_speed, self.max_speed)
        self.cmd_t = rospy.Time.now()

    def cb_tape(self, msg: Vector3Stamped):
        self.tape = msg
        self.tape_t = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

    def _reset_stats(self):
        self.ct_sq = 0.0
        self.ct_max = 0.0
        self.hd_sq = 0.0
        self.hd_max = 0.0
        self.stat_n = 0

    def maybe_reconfigure(self, now):
        if not self.live_reconfig or (now - self._last_reconfig).to_sec() < self.reconfig_period:
            return
        self._last_reconfig = now
        for name in self._tunables:
            new = float(rospy.get_param("~" + name, getattr(self, name)))
            if abs(new - getattr(self, name)) > 1e-9:
                rospy.loginfo("[line_keeper] %s: %.4f -> %.4f", name, getattr(self, name), new)
                setattr(self, name, new)

    def accumulate_stats(self, ct, hd):
        # Track the RAW (pre-deadband) errors while rolling -> true achieved accuracy.
        self.ct_sq += ct * ct
        self.hd_sq += hd * hd
        self.ct_max = max(self.ct_max, abs(ct))
        self.hd_max = max(self.hd_max, abs(hd))
        self.stat_n += 1

    def maybe_log_stats(self, now):
        if self.stats_period <= 0.0 or (now - self._last_stats).to_sec() < self.stats_period:
            return
        self._last_stats = now
        if self.stat_n > 0:
            ct_rms = math.sqrt(self.ct_sq / self.stat_n)
            hd_rms = math.sqrt(self.hd_sq / self.stat_n)
            rospy.loginfo("[line_keeper] lateral hold: rms=%.1fmm max=%.1fmm | "
                          "heading rms=%.2fdeg max=%.2fdeg  (%d samples/%.0fs, band=%.0fmm)",
                          ct_rms * 1000.0, self.ct_max * 1000.0,
                          math.degrees(hd_rms), math.degrees(self.hd_max),
                          self.stat_n, self.stats_period, self.deadband_ct * 1000.0)
        self._reset_stats()

    def step(self, now, dt):
        v = self.cmd_vx if (now - self.cmd_t).to_sec() <= self.cmd_timeout else 0.0

        tape_ok = (self.tape is not None
                   and (now - self.tape_t).to_sec() <= self.tape_timeout
                   and self.tape.vector.z >= self.min_confidence)

        if not tape_ok:
            if self.stop_on_tape_loss:
                v = 0.0
            self.wz = 0.0
            rospy.logwarn_throttle(2.0, "[line_keeper] tape lost/low-confidence -> %s",
                                   "STOP" if self.stop_on_tape_loss else "coast, no steer")
            self.publish(v, 0.0)
            return

        # Only steer while actually travelling (steering is a no-op at v=0 and a
        # nonzero wz there would trigger spin mode).
        if abs(v) < self.v_min:
            self.wz = 0.0
            self.publish(v, 0.0)
            return

        ct = self.tape.vector.x
        hd = self.tape.vector.y
        self.accumulate_stats(ct, hd)   # raw errors -> achieved-accuracy meter
        if abs(ct) < self.deadband_ct:
            ct = 0.0
        if abs(hd) < self.deadband_hd:
            hd = 0.0

        wz_des = self.sign * (self.k_heading * hd + self.k_cross * ct * math.copysign(1.0, v))
        wz_des = clamp(wz_des, -self.max_wz, self.max_wz)

        # Slew-rate limit for smoothness at slow speed.
        dwz = clamp(wz_des - self.wz, -self.wz_slew * dt, self.wz_slew * dt)
        self.wz = clamp(self.wz + dwz, -self.max_wz, self.max_wz)
        self.publish(v, self.wz)

    def publish(self, v, wz):
        t = Twist()
        t.linear.x = v
        t.linear.y = 0.0    # keep Ackermann mode
        t.angular.z = wz
        self.pub.publish(t)

    def spin(self):
        r = rospy.Rate(self.rate_hz)
        last = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - last).to_sec()
            last = now
            if dt <= 0.0:
                dt = 1.0 / self.rate_hz
            self.maybe_reconfigure(now)
            self.step(now, dt)
            self.maybe_log_stats(now)
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("line_keeper")
    LineKeeper().spin()
