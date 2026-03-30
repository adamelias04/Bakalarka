#!/usr/bin/env python3

import math

import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker


class SafeZoneGoalRouter:
    def __init__(self):
        self.input_goal_topic = rospy.get_param("~input_goal_topic", "/move_base_simple/goal")
        self.output_goal_topic = rospy.get_param("~output_goal_topic", "/managed_move_base_simple/goal")
        self.path_topic = rospy.get_param("~path_topic", "/predicted_obstacle_path")
        self.pose_topic = rospy.get_param("~pose_topic", "/detected_obstacle_pose")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.local_costmap_topic = rospy.get_param("~local_costmap_topic", "/move_base/local_costmap/costmap")
        self.output_frame = rospy.get_param("~output_frame", "odom")

        self.use_pose_as_start = rospy.get_param("~use_pose_as_start", True)
        self.pose_staleness_tolerance = rospy.get_param("~pose_staleness_tolerance", 0.35)
        self.pose_path_deviation_tolerance = rospy.get_param("~pose_path_deviation_tolerance", 0.45)
        self.velocity_epsilon = rospy.get_param("~velocity_epsilon", 0.03)
        self.max_path_length = rospy.get_param("~max_path_length", 3.24)
        self.obstacle_radius = rospy.get_param("~obstacle_radius", 0.22)
        self.corridor_half_width = rospy.get_param("~corridor_half_width", 0.45)
        self.side_extra_width = rospy.get_param("~side_extra_width", 0.30)
        self.visualization_clearance = rospy.get_param("~visualization_clearance", 0.55)
        self.rear_zone_length_scale = rospy.get_param("~rear_zone_length_scale", 2.25)
        self.front_zone_length_scale = rospy.get_param("~front_zone_length_scale", 1.0)
        self.robot_radius = rospy.get_param("~robot_radius", 0.50)
        self.hazard_margin = rospy.get_param("~hazard_margin", 0.20)
        self.rear_zone_preference = rospy.get_param("~rear_zone_preference", 1.25)
        self.front_zone_penalty = rospy.get_param("~front_zone_penalty", 0.40)
        self.rear_extra_distance_tolerance = rospy.get_param("~rear_extra_distance_tolerance", 2.0)
        self.subgoal_reached_distance = rospy.get_param("~subgoal_reached_distance", 0.70)
        self.goal_reached_distance = rospy.get_param("~goal_reached_distance", 0.25)
        self.costmap_block_threshold = rospy.get_param("~costmap_block_threshold", 85)
        self.costmap_check_step = rospy.get_param("~costmap_check_step", 0.10)
        self.timer_rate = rospy.get_param("~timer_rate", 10.0)
        self.publish_debug_marker = rospy.get_param("~publish_debug_marker", True)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.goal_pub = rospy.Publisher(self.output_goal_topic, PoseStamped, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("~debug_marker", Marker, queue_size=4, latch=True)

        rospy.Subscriber(self.input_goal_topic, PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.local_costmap_topic, OccupancyGrid, self.costmap_callback, queue_size=1)

        self.latest_path = None
        self.latest_pose = None
        self.latest_robot_pose = None
        self.latest_costmap = None

        self.final_goal = None
        self.active_target = None
        self.active_mode = "idle"
        self.locked_subgoal = None
        self.locked_subgoal_name = None
        self.locked_deep_subgoal = None

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.timer_rate, 1.0)), self.on_timer)

    @staticmethod
    def point_xy(msg):
        return (msg.x, msg.y)

    @staticmethod
    def distance(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def clamp01(v):
        return max(0.0, min(1.0, v))

    @staticmethod
    def offset(point, direction, distance):
        return (point[0] + direction[0] * distance, point[1] + direction[1] * distance)

    @staticmethod
    def yaw_to_quaternion(yaw):
        half = 0.5 * yaw
        return math.sin(half), math.cos(half)

    def world_to_costmap(self, point):
        if self.latest_costmap is None:
            return None

        origin = self.latest_costmap.info.origin.position
        resolution = self.latest_costmap.info.resolution
        width = self.latest_costmap.info.width
        height = self.latest_costmap.info.height

        mx = int(math.floor((point[0] - origin.x) / resolution))
        my = int(math.floor((point[1] - origin.y) / resolution))
        if mx < 0 or my < 0 or mx >= width or my >= height:
            return None

        return mx, my

    def costmap_value_at(self, point):
        index = self.world_to_costmap(point)
        if index is None or self.latest_costmap is None:
            return None

        mx, my = index
        flat_index = my * self.latest_costmap.info.width + mx
        if flat_index < 0 or flat_index >= len(self.latest_costmap.data):
            return None

        return self.latest_costmap.data[flat_index]

    def line_blocked_in_costmap(self, start, end):
        if self.latest_costmap is None:
            return False

        distance = self.distance(start, end)
        steps = max(1, int(math.ceil(distance / max(self.costmap_check_step, 1e-3))))
        for i in range(steps + 1):
            ratio = float(i) / float(steps)
            point = (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
            value = self.costmap_value_at(point)
            if value is None:
                continue
            if value >= self.costmap_block_threshold:
                return True

        return False

    def goal_callback(self, msg):
        goal = self.transform_pose(msg, self.output_frame)
        if goal is None:
            rospy.logwarn_throttle(1.0, "SafeZoneGoalRouter: nepodarilo sa transformovat goal do %s", self.output_frame)
            return

        self.final_goal = goal
        self.locked_subgoal = None
        self.locked_subgoal_name = None
        self.locked_deep_subgoal = None
        self.active_mode = "idle"
        self.active_target = None
        rospy.loginfo("SafeZoneGoalRouter: novy final goal prijaty")
        self.route_goal(force_publish=True)

    def path_callback(self, msg):
        self.latest_path = msg

    def pose_callback(self, msg):
        self.latest_pose = msg

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        transformed = self.transform_pose(pose, self.output_frame)
        if transformed is not None:
            self.latest_robot_pose = transformed

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def transform_pose(self, pose, target_frame):
        if pose.header.frame_id == target_frame:
            return pose

        try:
            return self.tf_buffer.transform(pose, target_frame, rospy.Duration(0.2))
        except Exception:
            return None

    def get_robot_xy(self):
        if self.latest_robot_pose is None:
            return None
        return (
            self.latest_robot_pose.pose.position.x,
            self.latest_robot_pose.pose.position.y,
        )

    def get_goal_xy(self):
        if self.final_goal is None:
            return None
        return (
            self.final_goal.pose.position.x,
            self.final_goal.pose.position.y,
        )

    def has_fresh_observation(self):
        if self.latest_path is None or not self.latest_path.poses or self.latest_pose is None:
            return False

        path_stamp = self.latest_path.header.stamp
        pose_stamp = self.latest_pose.header.stamp
        if not path_stamp.is_zero() and not pose_stamp.is_zero():
            if abs((path_stamp - pose_stamp).to_sec()) > self.pose_staleness_tolerance:
                return False

        path_start = self.latest_path.poses[0].pose.position
        pose_start = self.latest_pose.pose.position
        if self.distance((path_start.x, path_start.y), (pose_start.x, pose_start.y)) > self.pose_path_deviation_tolerance:
            return False

        return True

    def select_start_point(self):
        if self.latest_path is None or not self.latest_path.poses:
            return None

        if self.use_pose_as_start and self.latest_pose is not None and self.has_fresh_observation():
            return (
                self.latest_pose.pose.position.x,
                self.latest_pose.pose.position.y,
            )

        start = self.latest_path.poses[0].pose.position
        return (start.x, start.y)

    def build_geometry(self):
        if self.latest_path is None or not self.latest_path.poses:
            return None

        start = self.select_start_point()
        if start is None:
            return None

        end = start
        accumulated = 0.0
        prev = start

        for pose_stamped in self.latest_path.poses:
            cur = (pose_stamped.pose.position.x, pose_stamped.pose.position.y)
            seg = self.distance(prev, cur)
            if seg < 1e-6:
                prev = cur
                continue

            if accumulated + seg >= self.max_path_length:
                remain = max(0.0, self.max_path_length - accumulated)
                ratio = remain / seg
                end = (
                    prev[0] + ratio * (cur[0] - prev[0]),
                    prev[1] + ratio * (cur[1] - prev[1]),
                )
                accumulated = self.max_path_length
                break

            end = cur
            accumulated += seg
            prev = cur

        dir_vec = (end[0] - start[0], end[1] - start[1])
        length = math.hypot(dir_vec[0], dir_vec[1])
        if length < self.velocity_epsilon:
            return None

        direction = (dir_vec[0] / length, dir_vec[1] / length)
        normal = (-direction[1], direction[0])
        return {
            "start": start,
            "end": end,
            "dir": direction,
            "normal": normal,
            "length": length,
        }

    def compute_zone(self, near_point, direction, length):
        return {
            "near": near_point,
            "dir": direction,
            "length": max(length, 1e-3),
        }

    def compute_rear_zone(self, geometry):
        near = self.offset(
            geometry["start"],
            geometry["dir"],
            -(self.obstacle_radius + self.visualization_clearance),
        )
        direction = (-geometry["dir"][0], -geometry["dir"][1])
        length = geometry["length"] * self.rear_zone_length_scale
        return self.compute_zone(near, direction, length)

    def compute_front_zone(self, geometry):
        near = self.offset(geometry["end"], geometry["dir"], self.visualization_clearance)
        length = geometry["length"] * self.front_zone_length_scale
        return self.compute_zone(near, geometry["dir"], length)

    def zone_center(self, zone):
        return self.offset(zone["near"], zone["dir"], 0.5 * zone["length"])

    def rear_zone_target(self, geometry):
        rear_zone = self.compute_rear_zone(geometry)
        target_distance = max(0.90 * rear_zone["length"], 1.1)
        target_distance = min(target_distance, max(rear_zone["length"] - 0.10, 0.05))
        return self.offset(rear_zone["near"], rear_zone["dir"], target_distance)

    def rear_zone_entry_target(self, geometry):
        rear_zone = self.compute_rear_zone(geometry)
        target_distance = max(0.30 * rear_zone["length"], 0.45)
        target_distance = min(target_distance, max(rear_zone["length"] - 0.15, 0.05))
        return self.offset(rear_zone["near"], rear_zone["dir"], target_distance)

    def is_robot_behind_obstacle(self, robot_xy, geometry):
        to_robot = (
            robot_xy[0] - geometry["start"][0],
            robot_xy[1] - geometry["start"][1],
        )
        rear_progress = to_robot[0] * (-geometry["dir"][0]) + to_robot[1] * (-geometry["dir"][1])
        return rear_progress >= (self.obstacle_radius + self.visualization_clearance)

    def point_to_segment_distance(self, p, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-9:
            return self.distance(p, a)
        t = self.clamp01(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq)
        proj = (a[0] + t * dx, a[1] + t * dy)
        return self.distance(p, proj)

    def segment_to_segment_distance(self, a0, a1, b0, b1):
        return min(
            self.point_to_segment_distance(a0, b0, b1),
            self.point_to_segment_distance(a1, b0, b1),
            self.point_to_segment_distance(b0, a0, a1),
            self.point_to_segment_distance(b1, a0, a1),
        )

    def direct_path_blocked(self, robot_xy, goal_xy, geometry):
        if self.line_blocked_in_costmap(robot_xy, goal_xy):
            return True

        rear_zone = self.compute_rear_zone(geometry)
        front_zone = self.compute_front_zone(geometry)
        middle_start = rear_zone["near"]
        middle_end = front_zone["near"]

        body_radius = self.obstacle_radius + self.robot_radius + self.hazard_margin
        corridor_radius = (
            self.corridor_half_width + self.side_extra_width + self.robot_radius + self.hazard_margin
        )

        if self.point_to_segment_distance(geometry["start"], robot_xy, goal_xy) <= body_radius:
            return True

        if self.segment_to_segment_distance(robot_xy, goal_xy, middle_start, middle_end) <= corridor_radius:
            return True

        return False

    def choose_subgoal(self, robot_xy, goal_xy, geometry):
        return "rear_deep", self.rear_zone_target(geometry)

    def make_pose(self, xy, template_pose, stamp=None, yaw=None):
        pose = PoseStamped()
        pose.header.stamp = stamp if stamp is not None else rospy.Time.now()
        pose.header.frame_id = self.output_frame
        pose.pose = template_pose.pose
        pose.pose.position.x = xy[0]
        pose.pose.position.y = xy[1]
        pose.pose.position.z = 0.0
        if yaw is not None:
            qz, qw = self.yaw_to_quaternion(yaw)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
        return pose

    def publish_target(self, pose, mode):
        self.goal_pub.publish(pose)
        self.active_target = pose
        self.active_mode = mode
        rospy.loginfo(
            "SafeZoneGoalRouter: publikujem %s x=%.3f y=%.3f",
            mode,
            pose.pose.position.x,
            pose.pose.position.y,
        )
        self.publish_debug(mode)

    def publish_debug(self, mode):
        if not self.publish_debug_marker:
            return

        marker = Marker()
        marker.header.frame_id = self.output_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "safe_zone_goal_router"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.24
        marker.scale.y = 0.24
        marker.scale.z = 0.24

        if self.active_target is None:
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
            return

        marker.pose.position = self.active_target.pose.position
        if mode == "subgoal":
            marker.color.r = 0.1
            marker.color.g = 0.95
            marker.color.b = 0.2
            marker.color.a = 0.9
        else:
            marker.color.r = 0.1
            marker.color.g = 0.4
            marker.color.b = 0.95
            marker.color.a = 0.8

        self.marker_pub.publish(marker)

    def pose_changed(self, a, b, tol=0.05):
        if a is None or b is None:
            return True
        pa = a.pose.position
        pb = b.pose.position
        return self.distance((pa.x, pa.y), (pb.x, pb.y)) > tol

    def route_goal(self, force_publish=False):
        if self.final_goal is None:
            return

        robot_xy = self.get_robot_xy()
        goal_xy = self.get_goal_xy()
        if goal_xy is None:
            return

        if robot_xy is None:
            if force_publish or self.active_mode != "final" or self.pose_changed(self.active_target, self.final_goal):
                self.publish_target(self.final_goal, "final")
            return

        if self.distance(robot_xy, goal_xy) <= self.goal_reached_distance:
            self.active_mode = "idle"
            self.locked_subgoal = None
            self.locked_subgoal_name = None
            self.locked_deep_subgoal = None
            self.active_target = None
            self.publish_debug("idle")
            return

        if self.locked_subgoal is not None:
            subgoal_xy = (
                self.locked_subgoal.pose.position.x,
                self.locked_subgoal.pose.position.y,
            )

            geometry = self.build_geometry()
            if (
                self.locked_subgoal_name == "rear_deep"
                and geometry is not None
                and self.is_robot_behind_obstacle(robot_xy, geometry)
                and not self.line_blocked_in_costmap(robot_xy, goal_xy)
            ):
                self.locked_subgoal = None
                self.locked_subgoal_name = None
                self.locked_deep_subgoal = None
                rospy.loginfo("SafeZoneGoalRouter: handoff na final goal po prejdeni za kocku")
                if force_publish or self.active_mode != "final" or self.pose_changed(self.active_target, self.final_goal):
                    self.publish_target(self.final_goal, "final")
                return

            if self.distance(robot_xy, subgoal_xy) <= self.subgoal_reached_distance:
                if self.locked_subgoal_name == "rear_deep":
                    self.locked_subgoal = None
                    self.locked_subgoal_name = None
                    self.locked_deep_subgoal = None
                    if force_publish or self.active_mode != "final" or self.pose_changed(self.active_target, self.final_goal):
                        self.publish_target(self.final_goal, "final")
                    return
                else:
                    self.locked_subgoal = None
                    self.locked_subgoal_name = None
                    self.locked_deep_subgoal = None

            if self.locked_subgoal is not None and (
                force_publish or self.active_mode != "subgoal" or self.pose_changed(self.active_target, self.locked_subgoal)
            ):
                self.publish_target(self.locked_subgoal, "subgoal")
                return

            if self.locked_subgoal is not None:
                return

        geometry = self.build_geometry()
        if geometry is None or not self.direct_path_blocked(robot_xy, goal_xy, geometry):
            if force_publish or self.active_mode != "final" or self.pose_changed(self.active_target, self.final_goal):
                self.publish_target(self.final_goal, "final")
            return

        if self.locked_subgoal is None:
            deep_xy = self.rear_zone_target(geometry)
            if deep_xy is None:
                if force_publish or self.active_mode != "final" or self.pose_changed(self.active_target, self.final_goal):
                    self.publish_target(self.final_goal, "final")
                return
            heading = math.atan2(deep_xy[1] - robot_xy[1], deep_xy[0] - robot_xy[0])
            self.locked_deep_subgoal = self.make_pose(deep_xy, self.final_goal, yaw=heading)
            self.locked_subgoal = self.locked_deep_subgoal
            self.locked_subgoal_name = "rear_deep"
            rospy.loginfo("SafeZoneGoalRouter: zamykam rear_deep manever")

        if force_publish or self.active_mode != "subgoal" or self.pose_changed(self.active_target, self.locked_subgoal):
            self.publish_target(self.locked_subgoal, "subgoal")

    def on_timer(self, _event):
        self.route_goal(force_publish=False)


if __name__ == "__main__":
    rospy.init_node("safe_zone_goal_router")
    SafeZoneGoalRouter()
    rospy.spin()
