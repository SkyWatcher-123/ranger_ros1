#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel pick-and-place for the Ranger-Mini + xArm6 mobile manipulator (MoveIt).

Flow (see the task spec):
  A  prepare : read yaml, receive a place goal (xg,yg,zg) in world.
  B  pick    : B1 base->fixed X (locks after settle) | B2 scan pose (turn left)
               | B3 fetch /tag0 TF, cartesian to it | B4 vacuum ON + attach panel
               | B5 cartesian out to a safe joint pose | B6 cartesian to hold_home.
  C  place   : C1 base->(xg-offset) | C2 hold_home->hold_place | C3 cartesian to a
               camera stand-off = tag-LOCAL Z +15cm (NOT world Z) | C4 fetch /tag1,
               cartesian to it | C5 place: vacuum OFF + detach panel | C6 back out
               (no collision check) then cartesian to hold_home.

Design choices (kept deliberately simple / robust):
  * All arm motions are straight-line Cartesian (compute_cartesian_path) or a
    Cartesian move to the FK pose of a named/joint "hold" configuration -> good in
    tight spaces. Vel/acc scaled to 0.05.
  * Base moves reuse the existing /ranger/set_base_x service and wait for
    /ranger/at_target_x then /ranger/base_pose_locked (the settle-and-latch).
  * Vacuum gripper is toggled with the exact rosservice call from the spec.
  * No custom messages -> nothing to build. Goal comes in on a geometry_msgs/Point
    topic (~panel_goal) and the cycle is triggered by a std_srvs/Trigger service
    (~place_panel); or pass --xg/--yg/--zg to run once.
  * Test phase: every action waits for the human to type 'y'<enter> (unless --auto).
    --sim skips the two visual-confirmation moves (B2, C3).

Run (no package build needed):
  rosrun ranger_bringup pick_place_panel.py _config:=$(rospack find ranger_bringup)/config/pick_place_params.yaml
  # or:  python3 pick_place_panel.py --config /path/pick_place_params.yaml --xg 3.6 --yg 0.0 --zg 0.8
"""

import argparse
import copy
import subprocess
import sys

import numpy as np
import rospy
import tf2_ros
from tf import transformations as tft

import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped, Point
from std_msgs.msg import Bool, Header
from std_srvs.srv import Trigger, TriggerResponse
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import PositionIKRequest


# ----------------------------- pose / matrix helpers -----------------------------
def mat_from_pose(p):
    M = tft.quaternion_matrix([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    M[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return M


def pose_from_mat(M):
    q = tft.quaternion_from_matrix(M)
    p = Pose()
    p.position.x, p.position.y, p.position.z = M[0, 3], M[1, 3], M[2, 3]
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


def mat_xyzrpy(vals):
    x, y, z, R, P, Y = vals
    M = tft.euler_matrix(R, P, Y)
    M[:3, 3] = [x, y, z]
    return M


def mat_from_tf(tr):
    t, q = tr.transform.translation, tr.transform.rotation
    M = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    M[:3, 3] = [t.x, t.y, t.z]
    return M


# --------------------------------- main class -----------------------------------
class PickPlace(object):
    def __init__(self, cfg, sim=False, auto=False):
        self.cfg = cfg
        self.sim = sim
        self.auto = auto

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.group_name = cfg.get("group_name", "xarm6")
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        self.planning_frame = cfg.get("planning_frame") or self.group.get_planning_frame()
        self.eef_link = cfg.get("eef_link") or self.group.get_end_effector_link()
        self.world_frame = cfg.get("world_frame", "world")
        self.camera_frame = cfg.get("camera_frame", "xarm_color_frame")
        self.joint_names = self.group.get_active_joints()

        self.vscale = float(cfg.get("vel_scale", 0.05))
        self.ascale = float(cfg.get("acc_scale", 0.05))
        self.min_fraction = float(cfg.get("min_cartesian_fraction", 0.9))
        self.group.set_max_velocity_scaling_factor(self.vscale)
        self.group.set_max_acceleration_scaling_factor(self.ascale)

        # base
        self.enable_base = bool(cfg.get("enable_base", True))
        self.base_tol = float(cfg.get("base_tolerance", 0.03))
        self.settle_timeout = float(cfg.get("settle_timeout", 25.0))
        self.at_target = False
        self.pose_locked = False
        rospy.Subscriber("/ranger/at_target_x", Bool, lambda m: setattr(self, "at_target", m.data))
        rospy.Subscriber("/ranger/base_pose_locked", Bool, lambda m: setattr(self, "pose_locked", m.data))

        # panel bookkeeping
        self.panel_size = list(cfg.get("panel_size", [0.2, 0.75, 0.005]))
        self.panel_off = list(cfg.get("panel_attach_offset", [0.0, 0.0, -0.0025]))
        self.touch_links = list(cfg.get("touch_links") or [self.eef_link])
        self.panel_idx = int(cfg.get("start_panel_index", 0))

        # TF + FK/IK services (provided by move_group; no build needed)
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        rospy.wait_for_service("/compute_fk", timeout=15.0)
        rospy.wait_for_service("/compute_ik", timeout=15.0)
        self.fk_srv = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

        self.place_standoff = None   # cached C3 TCP pose (reused when backing out in C6)
        rospy.loginfo("[pick_place] group=%s planning_frame=%s eef=%s camera=%s sim=%s auto=%s",
                      self.group_name, self.planning_frame, self.eef_link, self.camera_frame,
                      self.sim, self.auto)

    # ------------------------------ small utilities ------------------------------
    def confirm(self, msg):
        rospy.loginfo("[step] %s", msg)
        if self.auto:
            return True
        if not sys.stdin or not sys.stdin.isatty():
            rospy.logerr("interactive confirm needs a terminal -- run via rosrun, or set auto:=true.")
            return False
        try:
            return input("       type 'y'<enter> to proceed: ").strip().lower() == "y"
        except EOFError:
            return False

    def world_to_planning(self, M_world):
        """Express a world-frame 4x4 in the arm planning frame (base may be elsewhere)."""
        if self.world_frame == self.planning_frame:
            return M_world
        try:
            tr = self.tf_buf.lookup_transform(self.planning_frame, self.world_frame,
                                              rospy.Time(0), rospy.Duration(3.0))
            return mat_from_tf(tr).dot(M_world)
        except Exception as e:
            rospy.logwarn("[pick_place] TF %s<-%s failed (%s); assuming same frame.",
                          self.planning_frame, self.world_frame, e)
            return M_world

    def tag_pose(self, tag_frame, nominal_world):
        """Tag pose in the planning frame: live TF if available, else nominal (world->planning)."""
        if not self.sim:
            try:
                tr = self.tf_buf.lookup_transform(self.planning_frame, tag_frame,
                                                  rospy.Time(0), rospy.Duration(3.0))
                rospy.loginfo("[pick_place] using live TF for %s", tag_frame)
                return mat_from_tf(tr)
            except Exception as e:
                rospy.logwarn("[pick_place] no live TF for %s (%s); using nominal.", tag_frame, e)
        return self.world_to_planning(mat_xyzrpy(nominal_world))

    # --------------------------------- base -------------------------------------
    def move_base_to_x(self, x):
        if not self.enable_base:
            rospy.logwarn("[base] disabled; skip move to x=%.3f", x)
            return True
        self.at_target = False
        self.pose_locked = False
        rospy.loginfo("[base] set_base_x -> %.3f (tol %.3f)", x, self.base_tol)
        try:
            subprocess.check_call(["rosservice", "call", "/ranger/set_base_x",
                                   "{target_x: %f, tolerance: %f}" % (x, self.base_tol)])
        except Exception as e:
            rospy.logerr("[base] set_base_x call failed: %s", e)
            return False
        t0 = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.pose_locked:
            if (rospy.Time.now() - t0).to_sec() > self.settle_timeout:
                rospy.logerr("[base] timed out waiting for arrival+lock")
                return False
            rate.sleep()
        rospy.loginfo("[base] arrived and locked.")
        return True

    # -------------------------------- FK / IK -----------------------------------
    def _header(self):
        h = Header()
        h.frame_id = self.planning_frame
        h.stamp = rospy.Time.now()
        return h

    def fk(self, joints):
        rs = self.robot.get_current_state()
        names, pos = list(rs.joint_state.name), list(rs.joint_state.position)
        for j, v in zip(self.joint_names, joints):
            pos[names.index(j)] = v
        rs.joint_state.position = pos
        resp = self.fk_srv(self._header(), [self.eef_link], rs)
        return resp.pose_stamped[0].pose

    def ik_exists(self, pose, label=""):
        req = PositionIKRequest()
        req.group_name = self.group_name
        req.robot_state = self.robot.get_current_state()
        req.avoid_collisions = True
        ps = PoseStamped()
        ps.header = self._header()
        ps.pose = pose
        req.pose_stamped = ps
        req.timeout = rospy.Duration(1.0)
        try:
            resp = self.ik_srv(req)
            ok = (resp.error_code.val == 1)
            rospy.loginfo("[IK] %s: %s", label, "solution exists" if ok else
                          "NO solution (code %d)" % resp.error_code.val)
            return ok
        except Exception as e:
            rospy.logwarn("[IK] %s check failed: %s", label, e)
            return False

    # ------------------------------- arm motion ---------------------------------
    def cart_move(self, target_pose, avoid_collisions=True, label=""):
        wp = [copy.deepcopy(target_pose)]
        try:                                             # ROS1 signature (with jump_threshold)
            plan, frac = self.group.compute_cartesian_path(wp, 0.005, 0.0, avoid_collisions)
        except TypeError:                                # newer signature (no jump_threshold)
            plan, frac = self.group.compute_cartesian_path(wp, 0.005, avoid_collisions=avoid_collisions)
        rospy.loginfo("[cart] %s: fraction=%.2f", label, frac)
        if frac < self.min_fraction and not self.confirm(
                "%s: low Cartesian fraction %.2f -- execute anyway?" % (label, frac)):
            return False
        plan = self.group.retime_trajectory(self.robot.get_current_state(), plan,
                                             self.vscale, self.ascale)
        return bool(self.group.execute(plan, wait=True))

    def resolve_joints(self, name):
        poses = self.cfg.get("poses", {})
        if name in poses:
            return list(poses[name])
        try:
            d = self.group.get_named_target_values(name)   # SRDF named state
            return [d[j] for j in self.joint_names]
        except Exception:
            raise RuntimeError("pose '%s' not in yaml 'poses' nor SRDF named targets" % name)

    def joint_move(self, joints, label=""):
        self.group.set_joint_value_target(list(joints))
        plan = self.group.plan()
        traj = plan[1] if isinstance(plan, tuple) else plan
        if not traj.joint_trajectory.points:
            rospy.logerr("[joint] %s: planning failed", label)
            return False
        traj = self.group.retime_trajectory(self.robot.get_current_state(), traj,
                                             self.vscale, self.ascale)
        return bool(self.group.execute(traj, wait=True))

    def goto_config(self, name, avoid_collisions=True):
        """Reach a named/joint 'hold' configuration via a Cartesian move to its FK pose,
        falling back to a joint move (with confirmation) if the straight line is infeasible."""
        joints = self.resolve_joints(name)
        pose = self.fk(joints)
        if self.cart_move(pose, avoid_collisions, label="cartesian->%s" % name):
            return True
        if self.confirm("cartesian to '%s' failed; do a joint-space move instead?" % name):
            return self.joint_move(joints, label=name)
        return False

    def tcp_target_from_tag(self, tag_M, override_key, approach_key):
        """Where the TCP should go to reach a tag. Default keeps the current TCP
        orientation and drives straight to the tag position (+ optional approach along
        the tag's normal). A full override (TCP pose expressed in the tag frame) may be
        given in yaml."""
        override = self.cfg.get(override_key)
        if override is not None:
            return pose_from_mat(tag_M.dot(mat_xyzrpy(override)))
        approach = float(self.cfg.get(approach_key, 0.0))
        cur = mat_from_pose(self.group.get_current_pose(self.eef_link).pose)
        M = np.eye(4)
        M[:3, :3] = cur[:3, :3]                                   # keep current orientation
        M[:3, 3] = tag_M[:3, 3] + approach * tag_M[:3, 2]         # tag position + approach along tag Z
        return pose_from_mat(M)

    # --------------------------------- panel ------------------------------------
    def attach_panel(self, name):
        ps = PoseStamped()
        ps.header.frame_id = self.eef_link
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = self.panel_off
        ps.pose.orientation.w = 1.0
        self.scene.attach_box(self.eef_link, name, ps, self.panel_size, self.touch_links)
        rospy.loginfo("[scene] attached %s (%.3fx%.3fx%.3f) at offset %s",
                      name, self.panel_size[0], self.panel_size[1], self.panel_size[2], self.panel_off)

    def detach_panel(self, name):
        self.scene.remove_attached_object(self.eef_link, name)
        rospy.sleep(0.5)
        self.scene.remove_world_object(name)
        rospy.loginfo("[scene] detached + removed %s", name)

    def vacuum(self, on):
        try:
            subprocess.check_call(["rosservice", "call", "/xarm6/vacuum_gripper_set", "1" if on else "0"])
            rospy.loginfo("[gripper] vacuum %s", "ON" if on else "OFF")
        except Exception as e:
            rospy.logerr("[gripper] vacuum_gripper_set failed: %s (check the service name/type)", e)

    # =============================== PICK (B) ===================================
    def pick(self):
        name = "panel_%d" % self.panel_idx
        rospy.loginfo("==== PICK %s ====", name)

        if not self.confirm("B1: move base to fixed pick X=%.3f" % self.cfg["pick_base_x"]):
            return False
        if not self.move_base_to_x(float(self.cfg["pick_base_x"])):
            return False

        if not self.sim:
            if not self.confirm("B2: move TCP to scan pose (turn to face the aisle)"):
                return False
            if not self.goto_config("scan_pose"):
                return False
        else:
            rospy.loginfo("[sim] skipping B2 scan pose")

        if not self.confirm("B3: fetch %s and move Cartesian to it" % self.cfg["pick_tag_frame"]):
            return False
        tagM = self.tag_pose(self.cfg["pick_tag_frame"], self.cfg["pickup_world"] + self.cfg.get("pick_tag_rpy", [0, 0, 0]))
        tgt = self.tcp_target_from_tag(tagM, "pick_tag_to_tcp", "pick_approach")
        self.ik_exists(tgt, "pick target")
        if not self.cart_move(tgt, True, "B3->tag"):
            return False

        if not self.confirm("B4: vacuum ON + attach %s" % name):
            return False
        self.vacuum(True)
        self.attach_panel(name)

        if not self.confirm("B5: Cartesian out to safe pick_out_pose"):
            return False
        if not self.goto_config("pick_out_pose"):
            return False

        if not self.confirm("B6: Cartesian to hold_home"):
            return False
        return self.goto_config("hold_home")

    # =============================== PLACE (C) ==================================
    def place(self, goal):
        xg, yg, zg = goal
        name = "panel_%d" % self.panel_idx
        rospy.loginfo("==== PLACE %s at world (%.3f, %.3f, %.3f) ====", name, xg, yg, zg)

        target_base_x = xg - float(self.cfg.get("base_x_place_offset", 0.20))
        if not self.confirm("C1: move base to X=%.3f (xg-%.2f)" % (target_base_x, self.cfg.get("base_x_place_offset", 0.20))):
            return False
        if not self.move_base_to_x(target_base_x):
            return False

        # Known place-tag pose = goal position + assumed tag orientation (world).
        place_tag_world = [xg, yg, zg] + list(self.cfg.get("place_tag_rpy", [0, 0, 0]))

        if not self.confirm("C2: hold_home -> hold_place (swing panel over base footprint)"):
            return False
        if not self.goto_config("hold_place"):
            return False

        # C3: Cartesian to a camera stand-off = tag LOCAL frame, +Z by place_scan_local_z.
        tagM = self.world_to_planning(mat_xyzrpy(place_tag_world))
        cam_off = mat_xyzrpy([0, 0, float(self.cfg.get("place_scan_local_z", 0.15))]
                             + list(self.cfg.get("place_cam_rpy_in_tag", [0, 0, 0])))
        cam_goal = tagM.dot(cam_off)                              # desired /camera pose (planning frame)
        T_cam_tcp = mat_from_tf(self.tf_buf.lookup_transform(     # tcp expressed in camera frame
            self.camera_frame, self.eef_link, rospy.Time(0), rospy.Duration(3.0)))
        self.place_standoff = pose_from_mat(cam_goal.dot(T_cam_tcp))
        self.ik_exists(self.place_standoff, "place stand-off (C3)")
        if not self.sim:
            if not self.confirm("C3: Cartesian to place-tag scan stand-off (cam at tag-local Z+%.2f)"
                                % float(self.cfg.get("place_scan_local_z", 0.15))):
                return False
            if not self.cart_move(self.place_standoff, True, "C3->standoff"):
                return False
        else:
            rospy.loginfo("[sim] skipping C3 visual stand-off move (pose still cached for C6)")

        if not self.confirm("C4: fetch %s and move Cartesian to it" % self.cfg["place_tag_frame"]):
            return False
        tag1 = self.tag_pose(self.cfg["place_tag_frame"], place_tag_world)
        tgt = self.tcp_target_from_tag(tag1, "place_tag_to_tcp", "place_approach")
        self.ik_exists(tgt, "place target (C4)")
        if not self.cart_move(tgt, True, "C4->tag"):
            return False

        if not self.confirm("C5: place -> vacuum OFF + detach %s" % name):
            return False
        self.vacuum(False)
        self.detach_panel(name)

        if not self.confirm("C6a: back out to stand-off (NO collision check)"):
            return False
        if not self.cart_move(self.place_standoff, False, "C6->standoff"):
            return False

        if not self.confirm("C6b: Cartesian to hold_home"):
            return False
        if not self.goto_config("hold_home"):
            return False

        self.panel_idx += 1                                      # panel_0 -> panel_1 -> ...
        rospy.loginfo("==== DONE. next index -> panel_%d ====", self.panel_idx)
        return True

    # --------------------------------- run --------------------------------------
    def run(self, goal):
        if not self.pick():
            rospy.logerr("PICK aborted."); return False
        if not self.place(goal):
            rospy.logerr("PLACE aborted."); return False
        return True


# ----------------------------------- main ---------------------------------------
def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--xg", type=float); ap.add_argument("--yg", type=float); ap.add_argument("--zg", type=float)
    args, _ = ap.parse_known_args()

    path = args.config or rospy.get_param("~config", None)
    if not path:
        rospy.logfatal("No config yaml. Pass --config or _config:=<path>."); sys.exit(1)
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    sim = args.sim or rospy.get_param("~sim", False)
    auto = args.auto or rospy.get_param("~auto", False)
    goal = None
    if args.xg is not None and args.yg is not None and args.zg is not None:
        goal = (args.xg, args.yg, args.zg)
    return cfg, sim, auto, goal


def main():
    rospy.init_node("pick_place_panel")
    cfg, sim, auto, goal = load_cfg()
    pp = PickPlace(cfg, sim=sim, auto=auto)

    if goal is not None:
        pp.run(goal)
        return

    # Service-triggered mode: goal arrives on ~panel_goal (Point), cycle on ~place_panel (Trigger).
    state = {"goal": cfg.get("default_goal")}
    rospy.Subscriber("~panel_goal", Point, lambda m: state.update(goal=(m.x, m.y, m.z)))

    def cb(_req):
        if state["goal"] is None:
            return TriggerResponse(False, "no goal: publish ~panel_goal (geometry_msgs/Point) first")
        ok = pp.run(tuple(state["goal"]))
        return TriggerResponse(ok, "cycle done" if ok else "cycle aborted")

    rospy.Service("~place_panel", Trigger, cb)
    rospy.loginfo("[pick_place] ready. Set goal: rostopic pub -1 %s/panel_goal geometry_msgs/Point \"{x: ..,y: ..,z: ..}\"; "
                  "then: rosservice call %s/place_panel", rospy.get_name(), rospy.get_name())
    rospy.spin()


if __name__ == "__main__":
    main()
