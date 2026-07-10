#!/usr/bin/env python3
"""
Publish known AprilTag poses (from rosparam tag_locations) as static TF frames.

Example:
  tag id 0 pose in odom  -> publish TF:  odom -> loc_0
  tag id 1 pose in odom  -> publish TF:  odom -> loc_1
"""

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


def _clean_frame(s: str) -> str:
    # TF2 convention: no leading '/'
    return s[1:] if s.startswith("/") else s


if __name__ == "__main__":
    rospy.init_node("publish_tag_locations_tf")

    parent_frame = _clean_frame(rospy.get_param("~parent_frame", "odom"))
    child_prefix = rospy.get_param("~child_prefix", "loc_")

    # Load from private param first, then global (matches your other node pattern)
    tag_list = rospy.get_param("~tag_locations", None)
    if tag_list is None:
        tag_list = rospy.get_param("/tag_locations", [])

    if not tag_list:
        rospy.logwarn("No tag_locations found. Did you load apriltag_localization.yaml via rosparam?")
        rospy.spin()

    br = tf2_ros.StaticTransformBroadcaster()
    tfs = []

    for item in tag_list:
        try:
            tid = int(item["id"])
            child_frame = _clean_frame(f"{child_prefix}{tid}")

            tfm = TransformStamped()
            tfm.header.stamp = rospy.Time.now()
            tfm.header.frame_id = parent_frame
            tfm.child_frame_id = child_frame

            tfm.transform.translation.x = float(item.get("x", 0.0))
            tfm.transform.translation.y = float(item.get("y", 0.0))
            tfm.transform.translation.z = float(item.get("z", 0.0))

            tfm.transform.rotation.x = float(item.get("qx", 0.0))
            tfm.transform.rotation.y = float(item.get("qy", 0.0))
            tfm.transform.rotation.z = float(item.get("qz", 0.0))
            tfm.transform.rotation.w = float(item.get("qw", 1.0))

            tfs.append(tfm)
        except Exception as e:
            rospy.logwarn(f"Skipping bad tag entry {item}: {e}")

    if not tfs:
        rospy.logwarn("No valid tag transforms to publish.")
        rospy.spin()

    br.sendTransform(tfs)
    rospy.loginfo(f"Published {len(tfs)} static tag TFs under parent '{parent_frame}' with prefix '{child_prefix}'")
    rospy.spin()
