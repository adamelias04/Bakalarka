#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>

namespace
{

struct Point2D
{
  double x;
  double y;
};

struct SweepGeometry
{
  Point2D start;
  Point2D end;
  Point2D dir;
  Point2D normal;
  double length;
};

struct ZoneGeometry
{
  Point2D near_point;
  Point2D dir;
  double length;
};

class SafeZoneGoalRouter
{
public:
  SafeZoneGoalRouter()
    : private_nh_("~"),
      tf_buffer_(ros::Duration(10.0)),
      tf_listener_(tf_buffer_),
      has_latest_path_(false),
      has_latest_pose_(false),
      has_latest_robot_pose_(false),
      has_latest_costmap_(false),
      has_final_goal_(false),
      has_active_target_(false),
      has_locked_subgoal_(false),
      has_locked_deep_subgoal_(false)
  {
    private_nh_.param<std::string>("input_goal_topic", input_goal_topic_, "/move_base_simple/goal");
    private_nh_.param<std::string>("output_goal_topic", output_goal_topic_, "/managed_move_base_simple/goal");
    private_nh_.param<std::string>("path_topic", path_topic_, "/predicted_obstacle_path");
    private_nh_.param<std::string>("pose_topic", pose_topic_, "/detected_obstacle_pose");
    private_nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
    private_nh_.param<std::string>("local_costmap_topic", local_costmap_topic_, "/move_base/local_costmap/costmap");
    private_nh_.param<std::string>("output_frame", output_frame_, "odom");

    private_nh_.param("use_pose_as_start", use_pose_as_start_, true);
    private_nh_.param("pose_staleness_tolerance", pose_staleness_tolerance_, 0.35);
    private_nh_.param("pose_path_deviation_tolerance", pose_path_deviation_tolerance_, 0.45);
    private_nh_.param("velocity_epsilon", velocity_epsilon_, 0.03);
    private_nh_.param("max_path_length", max_path_length_, 3.24);
    private_nh_.param("obstacle_radius", obstacle_radius_, 0.22);
    private_nh_.param("corridor_half_width", corridor_half_width_, 0.45);
    private_nh_.param("side_extra_width", side_extra_width_, 0.30);
    private_nh_.param("visualization_clearance", visualization_clearance_, 0.55);
    private_nh_.param("rear_zone_length_scale", rear_zone_length_scale_, 2.25);
    private_nh_.param("front_zone_length_scale", front_zone_length_scale_, 1.0);
    private_nh_.param("robot_radius", robot_radius_, 0.50);
    private_nh_.param("hazard_margin", hazard_margin_, 0.20);
    private_nh_.param("subgoal_reached_distance", subgoal_reached_distance_, 0.30);
    private_nh_.param("goal_reached_distance", goal_reached_distance_, 0.25);
    private_nh_.param("costmap_block_threshold", costmap_block_threshold_, 85);
    private_nh_.param("costmap_check_step", costmap_check_step_, 0.08);
    private_nh_.param("timer_rate", timer_rate_, 10.0);
    private_nh_.param("publish_debug_marker", publish_debug_marker_, true);

    goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(output_goal_topic_, 1, true);
    marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>("debug_marker", 4, true);

    goal_sub_ = nh_.subscribe(input_goal_topic_, 1, &SafeZoneGoalRouter::goalCallback, this);
    path_sub_ = nh_.subscribe(path_topic_, 1, &SafeZoneGoalRouter::pathCallback, this);
    pose_sub_ = nh_.subscribe(pose_topic_, 1, &SafeZoneGoalRouter::poseCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 1, &SafeZoneGoalRouter::odomCallback, this);
    costmap_sub_ = nh_.subscribe(local_costmap_topic_, 1, &SafeZoneGoalRouter::costmapCallback, this);

    timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(timer_rate_, 1.0)),
                             &SafeZoneGoalRouter::onTimer, this);
  }

private:
  static double clamp01(double value)
  {
    return std::max(0.0, std::min(1.0, value));
  }

  static double distance2d(const Point2D& a, const Point2D& b)
  {
    return std::hypot(b.x - a.x, b.y - a.y);
  }

  static Point2D offsetPoint(const Point2D& point, const Point2D& dir, double distance)
  {
    return Point2D{point.x + dir.x * distance, point.y + dir.y * distance};
  }

  bool transformPose(const geometry_msgs::PoseStamped& in, geometry_msgs::PoseStamped* out) const
  {
    if (out == nullptr)
      return false;

    if (in.header.frame_id == output_frame_)
    {
      *out = in;
      return true;
    }

    try
    {
      *out = tf_buffer_.transform(in, output_frame_, ros::Duration(0.2));
      return true;
    }
    catch (const tf2::TransformException&)
    {
      return false;
    }
  }

  Point2D poseToPoint(const geometry_msgs::PoseStamped& pose) const
  {
    return Point2D{pose.pose.position.x, pose.pose.position.y};
  }

  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    geometry_msgs::PoseStamped transformed;
    if (!transformPose(*msg, &transformed))
    {
      ROS_WARN_THROTTLE(1.0, "SafeZoneGoalRouter: nepodarilo sa transformovat goal do %s", output_frame_.c_str());
      return;
    }

    final_goal_ = transformed;
    has_final_goal_ = true;
    has_locked_subgoal_ = false;
    has_locked_deep_subgoal_ = false;
    active_mode_ = "idle";
    has_active_target_ = false;
    ROS_INFO("SafeZoneGoalRouter: novy final goal prijaty");
    routeGoal(true);
  }

  void pathCallback(const nav_msgs::Path::ConstPtr& msg)
  {
    latest_path_ = *msg;
    has_latest_path_ = !latest_path_.poses.empty();
  }

  void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    latest_pose_ = *msg;
    has_latest_pose_ = true;
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
  {
    geometry_msgs::PoseStamped pose;
    pose.header = msg->header;
    pose.pose = msg->pose.pose;
    geometry_msgs::PoseStamped transformed;
    if (transformPose(pose, &transformed))
    {
      latest_robot_pose_ = transformed;
      has_latest_robot_pose_ = true;
    }
  }

  void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
  {
    latest_costmap_ = *msg;
    has_latest_costmap_ = true;
  }

  bool hasFreshObservation() const
  {
    if (!has_latest_path_ || !has_latest_pose_ || latest_path_.poses.empty())
      return false;

    const ros::Time& path_stamp = latest_path_.header.stamp;
    const ros::Time& pose_stamp = latest_pose_.header.stamp;
    if (!path_stamp.isZero() && !pose_stamp.isZero())
    {
      if (std::fabs((path_stamp - pose_stamp).toSec()) > pose_staleness_tolerance_)
        return false;
    }

    const Point2D path_start{latest_path_.poses.front().pose.position.x, latest_path_.poses.front().pose.position.y};
    const Point2D pose_start{latest_pose_.pose.position.x, latest_pose_.pose.position.y};
    return distance2d(path_start, pose_start) <= pose_path_deviation_tolerance_;
  }

  Point2D selectStartPoint() const
  {
    Point2D start{latest_path_.poses.front().pose.position.x, latest_path_.poses.front().pose.position.y};
    if (use_pose_as_start_ && has_latest_pose_ && hasFreshObservation())
    {
      start.x = latest_pose_.pose.position.x;
      start.y = latest_pose_.pose.position.y;
    }
    return start;
  }

  bool buildGeometry(SweepGeometry* geometry) const
  {
    if (geometry == nullptr || !has_latest_path_ || latest_path_.poses.empty())
      return false;

    geometry->start = selectStartPoint();
    geometry->end = geometry->start;

    double accumulated = 0.0;
    Point2D prev = geometry->start;
    for (const auto& pose_stamped : latest_path_.poses)
    {
      Point2D cur{pose_stamped.pose.position.x, pose_stamped.pose.position.y};
      const double seg = distance2d(prev, cur);
      if (seg < 1e-6)
      {
        prev = cur;
        continue;
      }

      if (accumulated + seg >= max_path_length_)
      {
        const double remain = std::max(0.0, max_path_length_ - accumulated);
        const double ratio = remain / seg;
        geometry->end.x = prev.x + ratio * (cur.x - prev.x);
        geometry->end.y = prev.y + ratio * (cur.y - prev.y);
        accumulated = max_path_length_;
        break;
      }

      geometry->end = cur;
      accumulated += seg;
      prev = cur;
    }

    geometry->dir.x = geometry->end.x - geometry->start.x;
    geometry->dir.y = geometry->end.y - geometry->start.y;
    geometry->length = std::hypot(geometry->dir.x, geometry->dir.y);
    if (geometry->length < velocity_epsilon_)
      return false;

    geometry->dir.x /= geometry->length;
    geometry->dir.y /= geometry->length;
    geometry->normal.x = -geometry->dir.y;
    geometry->normal.y =  geometry->dir.x;
    return true;
  }

  ZoneGeometry computeRearZone(const SweepGeometry& geometry) const
  {
    ZoneGeometry zone;
    zone.near_point = offsetPoint(geometry.start, geometry.dir, -(obstacle_radius_ + visualization_clearance_));
    zone.dir = Point2D{-geometry.dir.x, -geometry.dir.y};
    zone.length = std::max(geometry.length * rear_zone_length_scale_, 1e-3);
    return zone;
  }

  Point2D rearZoneTarget(const SweepGeometry& geometry) const
  {
    const ZoneGeometry rear_zone = computeRearZone(geometry);
    double target_distance = std::max(0.90 * rear_zone.length, 1.1);
    target_distance = std::min(target_distance, std::max(rear_zone.length - 0.10, 0.05));
    return offsetPoint(rear_zone.near_point, rear_zone.dir, target_distance);
  }

  bool isRobotBehindObstacle(const Point2D& robot_xy, const SweepGeometry& geometry) const
  {
    const Point2D to_robot{robot_xy.x - geometry.start.x, robot_xy.y - geometry.start.y};
    const double rear_progress = to_robot.x * (-geometry.dir.x) + to_robot.y * (-geometry.dir.y);
    return rear_progress >= (obstacle_radius_ + visualization_clearance_);
  }

  double pointToSegmentDistance(const Point2D& p, const Point2D& a, const Point2D& b) const
  {
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-9)
      return distance2d(p, a);

    const double t = clamp01(((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq);
    const Point2D proj{a.x + t * dx, a.y + t * dy};
    return distance2d(p, proj);
  }

  double segmentToSegmentDistance(const Point2D& a0, const Point2D& a1,
                                  const Point2D& b0, const Point2D& b1) const
  {
    return std::min(
      std::min(pointToSegmentDistance(a0, b0, b1), pointToSegmentDistance(a1, b0, b1)),
      std::min(pointToSegmentDistance(b0, a0, a1), pointToSegmentDistance(b1, a0, a1)));
  }

  bool worldToCostmap(const Point2D& point, int* mx, int* my) const
  {
    if (!has_latest_costmap_ || mx == nullptr || my == nullptr)
      return false;

    const auto& info = latest_costmap_.info;
    const auto& origin = info.origin.position;
    *mx = static_cast<int>(std::floor((point.x - origin.x) / info.resolution));
    *my = static_cast<int>(std::floor((point.y - origin.y) / info.resolution));
    return *mx >= 0 && *my >= 0 &&
           *mx < static_cast<int>(info.width) &&
           *my < static_cast<int>(info.height);
  }

  bool costmapValueAt(const Point2D& point, int* value) const
  {
    if (value == nullptr)
      return false;

    int mx = 0;
    int my = 0;
    if (!worldToCostmap(point, &mx, &my))
      return false;

    const int index = my * static_cast<int>(latest_costmap_.info.width) + mx;
    if (index < 0 || index >= static_cast<int>(latest_costmap_.data.size()))
      return false;

    *value = static_cast<int>(latest_costmap_.data[static_cast<std::size_t>(index)]);
    return true;
  }

  bool lineBlockedInCostmap(const Point2D& start, const Point2D& end) const
  {
    if (!has_latest_costmap_)
      return false;

    const double distance = distance2d(start, end);
    const int steps = std::max(1, static_cast<int>(std::ceil(distance / std::max(costmap_check_step_, 1e-3))));
    for (int i = 0; i <= steps; ++i)
    {
      const double ratio = static_cast<double>(i) / static_cast<double>(steps);
      const Point2D point{
        start.x + ratio * (end.x - start.x),
        start.y + ratio * (end.y - start.y)
      };
      int value = 0;
      if (!costmapValueAt(point, &value))
        continue;
      if (value >= costmap_block_threshold_)
        return true;
    }

    return false;
  }

  bool directPathBlocked(const Point2D& robot_xy, const Point2D& goal_xy, const SweepGeometry& geometry) const
  {
    if (lineBlockedInCostmap(robot_xy, goal_xy))
      return true;

    const ZoneGeometry rear_zone = computeRearZone(geometry);
    const ZoneGeometry front_zone{
      offsetPoint(geometry.end, geometry.dir, visualization_clearance_),
      geometry.dir,
      std::max(geometry.length * front_zone_length_scale_, 1e-3)
    };

    const Point2D middle_start = rear_zone.near_point;
    const Point2D middle_end = front_zone.near_point;
    const double body_radius = obstacle_radius_ + robot_radius_ + hazard_margin_;
    const double corridor_radius = corridor_half_width_ + side_extra_width_ + robot_radius_ + hazard_margin_;

    if (pointToSegmentDistance(geometry.start, robot_xy, goal_xy) <= body_radius)
      return true;

    if (segmentToSegmentDistance(robot_xy, goal_xy, middle_start, middle_end) <= corridor_radius)
      return true;

    return false;
  }

  geometry_msgs::PoseStamped makePose(const Point2D& xy,
                                      const geometry_msgs::PoseStamped& template_pose,
                                      double yaw,
                                      bool use_custom_yaw) const
  {
    geometry_msgs::PoseStamped pose = template_pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = output_frame_;
    pose.pose.position.x = xy.x;
    pose.pose.position.y = xy.y;
    pose.pose.position.z = 0.0;
    if (use_custom_yaw)
    {
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, yaw);
      pose.pose.orientation = tf2::toMsg(q);
    }
    return pose;
  }

  void publishDebug(const std::string& mode)
  {
    if (!publish_debug_marker_)
      return;

    visualization_msgs::Marker marker;
    marker.header.frame_id = output_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "safe_zone_goal_router";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.24;
    marker.scale.y = 0.24;
    marker.scale.z = 0.24;

    if (!has_active_target_)
    {
      marker.action = visualization_msgs::Marker::DELETE;
      marker_pub_.publish(marker);
      return;
    }

    marker.pose.position = active_target_.pose.position;
    if (mode == "subgoal")
    {
      marker.color.r = 0.1;
      marker.color.g = 0.95;
      marker.color.b = 0.2;
      marker.color.a = 0.9;
    }
    else
    {
      marker.color.r = 0.1;
      marker.color.g = 0.4;
      marker.color.b = 0.95;
      marker.color.a = 0.8;
    }
    marker_pub_.publish(marker);
  }

  bool poseChanged(const geometry_msgs::PoseStamped& a, const geometry_msgs::PoseStamped& b, double tol = 0.05) const
  {
    const Point2D pa{a.pose.position.x, a.pose.position.y};
    const Point2D pb{b.pose.position.x, b.pose.position.y};
    return distance2d(pa, pb) > tol;
  }

  void publishTarget(const geometry_msgs::PoseStamped& pose, const std::string& mode)
  {
    goal_pub_.publish(pose);
    active_target_ = pose;
    has_active_target_ = true;
    active_mode_ = mode;
    ROS_INFO("SafeZoneGoalRouter: publikujem %s x=%.3f y=%.3f",
             mode.c_str(), pose.pose.position.x, pose.pose.position.y);
    publishDebug(mode);
  }

  void clearActiveTarget()
  {
    has_active_target_ = false;
    publishDebug("idle");
  }

  void routeGoal(bool force_publish)
  {
    if (!has_final_goal_)
      return;

    if (!has_latest_robot_pose_)
    {
      if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
        publishTarget(final_goal_, "final");
      return;
    }

    const Point2D robot_xy = poseToPoint(latest_robot_pose_);
    const Point2D goal_xy = poseToPoint(final_goal_);

    if (distance2d(robot_xy, goal_xy) <= goal_reached_distance_)
    {
      has_locked_subgoal_ = false;
      has_locked_deep_subgoal_ = false;
      active_mode_ = "idle";
      clearActiveTarget();
      return;
    }

    if (has_locked_subgoal_)
    {
      const Point2D subgoal_xy = poseToPoint(locked_subgoal_);

      SweepGeometry geometry;
      if (locked_subgoal_name_ == "rear_deep" &&
          buildGeometry(&geometry) &&
          isRobotBehindObstacle(robot_xy, geometry) &&
          !lineBlockedInCostmap(robot_xy, goal_xy))
      {
        has_locked_subgoal_ = false;
        has_locked_deep_subgoal_ = false;
        ROS_INFO("SafeZoneGoalRouter: handoff na final goal po prejdeni za kocku");
        if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
          publishTarget(final_goal_, "final");
        return;
      }

      if (distance2d(robot_xy, subgoal_xy) <= subgoal_reached_distance_)
      {
        has_locked_subgoal_ = false;
        has_locked_deep_subgoal_ = false;
        if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
          publishTarget(final_goal_, "final");
        return;
      }

      if (force_publish || !has_active_target_ || active_mode_ != "subgoal" || poseChanged(active_target_, locked_subgoal_))
        publishTarget(locked_subgoal_, "subgoal");
      return;
    }

    SweepGeometry geometry;
    if (!buildGeometry(&geometry) || !directPathBlocked(robot_xy, goal_xy, geometry))
    {
      if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
        publishTarget(final_goal_, "final");
      return;
    }

    const Point2D deep_xy = rearZoneTarget(geometry);
    const double heading = std::atan2(deep_xy.y - robot_xy.y, deep_xy.x - robot_xy.x);
    locked_deep_subgoal_ = makePose(deep_xy, final_goal_, heading, true);
    locked_subgoal_ = locked_deep_subgoal_;
    locked_subgoal_name_ = "rear_deep";
    has_locked_deep_subgoal_ = true;
    has_locked_subgoal_ = true;
    ROS_INFO("SafeZoneGoalRouter: zamykam rear_deep manever");

    if (force_publish || !has_active_target_ || active_mode_ != "subgoal" || poseChanged(active_target_, locked_subgoal_))
      publishTarget(locked_subgoal_, "subgoal");
  }

  void onTimer(const ros::TimerEvent&)
  {
    routeGoal(false);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::Publisher goal_pub_;
  ros::Publisher marker_pub_;
  ros::Subscriber goal_sub_;
  ros::Subscriber path_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber costmap_sub_;
  ros::Timer timer_;

  std::string input_goal_topic_;
  std::string output_goal_topic_;
  std::string path_topic_;
  std::string pose_topic_;
  std::string odom_topic_;
  std::string local_costmap_topic_;
  std::string output_frame_;
  std::string active_mode_;
  std::string locked_subgoal_name_;

  bool use_pose_as_start_;
  bool publish_debug_marker_;
  bool has_latest_path_;
  bool has_latest_pose_;
  bool has_latest_robot_pose_;
  bool has_latest_costmap_;
  bool has_final_goal_;
  bool has_active_target_;
  bool has_locked_subgoal_;
  bool has_locked_deep_subgoal_;

  double pose_staleness_tolerance_;
  double pose_path_deviation_tolerance_;
  double velocity_epsilon_;
  double max_path_length_;
  double obstacle_radius_;
  double corridor_half_width_;
  double side_extra_width_;
  double visualization_clearance_;
  double rear_zone_length_scale_;
  double front_zone_length_scale_;
  double robot_radius_;
  double hazard_margin_;
  double subgoal_reached_distance_;
  double goal_reached_distance_;
  double costmap_check_step_;
  double timer_rate_;

  int costmap_block_threshold_;

  nav_msgs::Path latest_path_;
  geometry_msgs::PoseStamped latest_pose_;
  geometry_msgs::PoseStamped latest_robot_pose_;
  nav_msgs::OccupancyGrid latest_costmap_;
  geometry_msgs::PoseStamped final_goal_;
  geometry_msgs::PoseStamped active_target_;
  geometry_msgs::PoseStamped locked_subgoal_;
  geometry_msgs::PoseStamped locked_deep_subgoal_;
};

}  // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "safe_zone_goal_router");
  SafeZoneGoalRouter router;
  ros::spin();
  return 0;
}
