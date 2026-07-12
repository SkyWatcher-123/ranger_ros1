#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect the straight yellow guide tape on the floor and report the Ranger's
lateral offset and heading error relative to it, for the line-keeping controller.

Pipeline (all geometry verified against the real loc_camera extrinsics):
  1. Threshold yellow in HSV on /loc_camera/color/image_raw.
  2. Inverse-perspective map (IPM) the mask to a metric bird's-eye view (BEV):
     using K (from camera_info) + TF base_link -> color-optical frame + the flat
     ground plane (z = -base_height in base_link), the ground->image projection is
     an exact homography. We warp the mask into a top-down, metric BEV window.
  3. Robust (RANSAC) line fit in BEV metric coordinates -> tape as Y = m*X + b.
        heading  = atan(m)                      [rad, tape dir vs base_link +x]
        lateral  = m*X_lookahead + b            [m, tape Y at a look-ahead point]
  4. Publish errors relative to a stored reference (captured when the robot is
     aligned on the tape, via the set_reference service -- the tape need NOT be
     centered under the robot/camera).

Output: geometry_msgs/Vector3Stamped on ~tape
    vector.x = cross_track_err = lateral  - lateral_ref   [m]   (+ = tape left of ref)
    vector.y = heading_err     = heading  - heading_ref   [rad] (wrapped)
    vector.z = confidence      in [0, 1]  (0 => not a valid detection this frame)

Why BEV/metric instead of raw image angle: the camera looks down at ~45 deg, so a
straight ground line's *image* angle is perspective-distorted and non-linear in
robot yaw. IPM removes perspective, giving errors in real meters/radians.
"""

import math
import threading

import numpy as np
import cv2

import rospy
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped
from std_srvs.srv import Trigger, TriggerResponse


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TapeLineDetector:
    def __init__(self):
        self.bridge = CvBridge()

        # --- Frames -----------------------------------------------------------
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        # Ground plane sits base_height below base_link (base_link is 0.346 m up).
        self.base_height = float(rospy.get_param("~base_height", 0.346))

        # --- BEV window (metric, base_link ground frame). Defaults match the
        #     measured visible footprint of the 45deg-down loc_camera. -----------
        self.x_near = float(rospy.get_param("~bev_x_near", 0.55))   # m ahead
        self.x_far = float(rospy.get_param("~bev_x_far", 1.20))
        self.y_right = float(rospy.get_param("~bev_y_right", -0.55))  # m (base_link +y = left)
        self.y_left = float(rospy.get_param("~bev_y_left", 0.35))
        self.ppm = float(rospy.get_param("~bev_pixels_per_meter", 300.0))
        # Look-ahead X at which the lateral offset is reported (front-axle-like).
        self.x_lookahead = float(rospy.get_param("~x_lookahead", 0.80))

        # --- Yellow HSV threshold (OpenCV H in 0..179). Tight-ish to reject the
        #     gray floor and any orange; tune on hardware with the debug image. --
        self.h_lo = int(rospy.get_param("~h_lo", 20))
        self.h_hi = int(rospy.get_param("~h_hi", 40))
        self.s_lo = int(rospy.get_param("~s_lo", 80))
        self.s_hi = int(rospy.get_param("~s_hi", 255))
        self.v_lo = int(rospy.get_param("~v_lo", 80))
        self.v_hi = int(rospy.get_param("~v_hi", 255))

        # --- Robustness / smoothing ------------------------------------------
        self.min_pixels = int(rospy.get_param("~min_pixels", 300))
        self.ransac_thresh_m = float(rospy.get_param("~ransac_thresh_m", 0.02))
        self.max_heading_deg = float(rospy.get_param("~max_heading_deg", 35.0))
        self.smooth_alpha = float(rospy.get_param("~smooth_alpha", 0.5))  # 1.0 = none
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        info_topic = rospy.get_param("~camera_info_topic", "/loc_camera/color/camera_info")
        image_topic = rospy.get_param("~image_topic", "/loc_camera/color/image_raw")

        # --- State ------------------------------------------------------------
        self.K = None
        self.optical_frame = None
        self.H_img_from_bev = None   # 3x3: BEV pixel -> source image pixel (for warp)
        self.have_geom = False
        self.lock = threading.Lock()

        self.lateral = None          # smoothed metric estimates
        self.heading = None
        self.lateral_ref = 0.0
        self.heading_ref = 0.0
        self.have_reference = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher("~tape", Vector3Stamped, queue_size=10)
        self.pub_debug = rospy.Publisher("~debug_image", Image, queue_size=1) if self.publish_debug else None

        self.sub_info = rospy.Subscriber(info_topic, CameraInfo, self.cb_info, queue_size=1)
        self.sub_img = rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1, buff_size=2 ** 24)

        self.srv = rospy.Service("~set_reference", Trigger, self.cb_set_reference)

        # BEV size (pixels). Row 0 (top) = far (x_far), bottom = near (x_near).
        self.bev_w = max(1, int(round((self.y_left - self.y_right) * self.ppm)))
        self.bev_h = max(1, int(round((self.x_far - self.x_near) * self.ppm)))

        rospy.loginfo("[tape_line_detector] BEV X[%.2f,%.2f] Y[%.2f,%.2f] @ %.0f px/m -> %dx%d; "
                      "waiting for camera_info on %s",
                      self.x_near, self.x_far, self.y_right, self.y_left, self.ppm,
                      self.bev_w, self.bev_h, info_topic)

    # ---- BEV <-> metric ground helpers --------------------------------------
    def bev_to_ground(self, col, row):
        """BEV pixel (col,row) -> ground (X,Y) in base_link [m]."""
        Y = self.y_left - (col / self.ppm)          # col 0 = y_left
        X = self.x_far - (row / self.ppm)           # row 0 = x_far
        return X, Y

    def ground_to_bev(self, X, Y):
        col = (self.y_left - Y) * self.ppm
        row = (self.x_far - X) * self.ppm
        return col, row

    def cb_info(self, msg: CameraInfo):
        if self.have_geom:
            return
        self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.optical_frame = msg.header.frame_id
        self.try_build_geometry()

    def try_build_geometry(self):
        """Compute the BEV<->image homography from K + TF + ground plane."""
        if self.K is None or not self.optical_frame:
            return
        try:
            tr = self.tf_buffer.lookup_transform(
                self.base_frame, self.optical_frame, rospy.Time(0), rospy.Duration(2.0))
        except Exception as e:
            rospy.logwarn_throttle(3.0, "[tape_line_detector] waiting for TF %s->%s: %s",
                                   self.base_frame, self.optical_frame, e)
            return

        t = tr.transform.translation
        q = tr.transform.rotation
        R_bl_opt = quat_to_R([q.x, q.y, q.z, q.w])   # base_link <- optical
        o_bl = np.array([t.x, t.y, t.z])             # optical origin in base_link
        R_opt_bl = R_bl_opt.T

        # Ground point (X,Y) on plane z=-base_height maps to optical coords linearly:
        #   P_opt = R_opt_bl * ([X,Y,-h] - o_bl)
        gz = -self.base_height
        M = np.column_stack([
            R_opt_bl @ np.array([1.0, 0.0, 0.0]),
            R_opt_bl @ np.array([0.0, 1.0, 0.0]),
            R_opt_bl @ np.array([0.0, 0.0, gz]) - R_opt_bl @ o_bl,
        ])
        H_img_from_ground = self.K @ M               # (X,Y,1)_ground -> (u,v,1)_pixel

        # (col,row)_BEV -> (X,Y,1)_ground : X = x_far - row/ppm ; Y = y_left - col/ppm
        G = np.array([
            [-1.0 / self.ppm, 0.0, self.y_left],     # Y from col
            [0.0, -1.0 / self.ppm, self.x_far],      # X from row
            [0.0, 0.0, 1.0],
        ])
        # G maps [col,row,1] -> [Y, X, 1]; reorder to [X, Y, 1]:
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
        bev_to_ground = swap @ G                     # [col,row,1] -> [X,Y,1]
        self.H_img_from_bev = H_img_from_ground @ bev_to_ground
        self.have_geom = True
        rospy.loginfo("[tape_line_detector] geometry ready (optical=%s, cam height=%.3f m).",
                      self.optical_frame, o_bl[2] + self.base_height)

    def cb_image(self, msg: Image):
        if not self.have_geom:
            self.try_build_geometry()
            if not self.have_geom:
                return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[tape_line_detector] cv_bridge: %s", e)
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.h_lo, self.s_lo, self.v_lo),
                                (self.h_hi, self.s_hi, self.v_hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        bev = cv2.warpPerspective(mask, self.H_img_from_bev, (self.bev_w, self.bev_h),
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                  borderValue=0)

        ok, lateral, heading, inliers, line_pts = self.fit_line_bev(bev)

        now = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        out = Vector3Stamped()
        out.header.stamp = now
        out.header.frame_id = self.base_frame

        if ok:
            with self.lock:
                if self.heading is None or self.smooth_alpha >= 0.999:
                    self.lateral, self.heading = lateral, heading
                else:
                    a = self.smooth_alpha
                    self.lateral += a * (lateral - self.lateral)
                    self.heading = wrap_to_pi(self.heading + a * wrap_to_pi(heading - self.heading))
                out.vector.x = self.lateral - self.lateral_ref
                out.vector.y = wrap_to_pi(self.heading - self.heading_ref)
                out.vector.z = min(1.0, inliers / float(max(self.min_pixels, 1)))
        else:
            out.vector.z = 0.0
        self.pub.publish(out)

        if self.pub_debug is not None:
            self.publish_debug_image(bev, ok, line_pts, out)

    def fit_line_bev(self, bev):
        ys, xs = np.nonzero(bev)                     # xs=col, ys=row
        n = xs.size
        if n < self.min_pixels:
            return False, 0.0, 0.0, n, None

        # Metric ground coords for each lit BEV pixel.
        X = self.x_far - ys / self.ppm
        Y = self.y_left - xs / self.ppm
        pts = np.column_stack([X, Y]).astype(np.float32)

        # RANSAC on Y = m*X + b (X well-conditioned: spans the window).
        best_in, best_mb = 0, None
        rng = np.random.default_rng(0)
        idx = np.arange(n)
        iters = 60
        for _ in range(iters):
            s = rng.choice(idx, 2, replace=False)
            x0, y0 = pts[s[0]]
            x1, y1 = pts[s[1]]
            if abs(x1 - x0) < 1e-6:
                continue
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            res = np.abs(pts[:, 1] - (m * pts[:, 0] + b))
            ninl = int(np.count_nonzero(res < self.ransac_thresh_m))
            if ninl > best_in:
                best_in, best_mb = ninl, (m, b)
        if best_mb is None or best_in < self.min_pixels:
            return False, 0.0, 0.0, best_in, None

        # Refit least-squares on inliers.
        m, b = best_mb
        res = np.abs(pts[:, 1] - (m * pts[:, 0] + b))
        inl = pts[res < self.ransac_thresh_m]
        A = np.column_stack([inl[:, 0], np.ones(len(inl))])
        m, b = np.linalg.lstsq(A, inl[:, 1], rcond=None)[0]

        heading = math.atan(m)
        if abs(math.degrees(heading)) > self.max_heading_deg:
            return False, 0.0, 0.0, best_in, None
        lateral = m * self.x_lookahead + b

        line_pts = ((self.x_near, m * self.x_near + b), (self.x_far, m * self.x_far + b))
        return True, lateral, heading, best_in, line_pts

    def cb_set_reference(self, req):
        with self.lock:
            if self.heading is None:
                return TriggerResponse(success=False, message="No valid tape detection yet.")
            self.lateral_ref = self.lateral
            self.heading_ref = self.heading
            self.have_reference = True
            msg = ("Reference set: lateral=%.3f m, heading=%.2f deg"
                   % (self.lateral_ref, math.degrees(self.heading_ref)))
        rospy.loginfo("[tape_line_detector] %s", msg)
        return TriggerResponse(success=True, message=msg)

    def publish_debug_image(self, bev, ok, line_pts, out):
        vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        if ok and line_pts is not None:
            (x0, y0), (x1, y1) = line_pts
            c0 = tuple(int(round(v)) for v in self.ground_to_bev(x0, y0))
            c1 = tuple(int(round(v)) for v in self.ground_to_bev(x1, y1))
            cv2.line(vis, c0, c1, (0, 255, 0), 2)
            txt = "ct=%.3fm hd=%.1fdeg c=%.2f" % (out.vector.x, math.degrees(out.vector.y), out.vector.z)
            cv2.putText(vis, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(vis, "NO TAPE", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))
        except Exception:
            pass


def quat_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.identity(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z),     s * (x * z + w * y)],
        [s * (x * y + w * z),     1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y),     s * (y * z + w * x),     1 - s * (x * x + y * y)],
    ])


if __name__ == "__main__":
    rospy.init_node("tape_line_detector")
    TapeLineDetector()
    rospy.spin()
