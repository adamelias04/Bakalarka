#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
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

// SafeZoneGoalRouter je teraz tenka vrstva nad move_base.
//
// Hlavna myslienka navigacie okolo dynamickej prekazky je nasledovna:
//   1) dynamic_obstacle_predictor detekuje prekazku a publikuje
//      predikovanu trajektoriu.
//   2) PredictedSweepRiskLayer (costmap plugin) zapise rizikove naklady
//      okolo tej trajektorie a sucasne publikuje soft via-points do
//      bezpecnych zon (pred/za prekazkou).
//   3) TEB local planner spracuva tie via-points ako mekke atraktory
//      a robota pritiahne do safe zony bez toho, aby sa za nimi nahanal.
//
// Preto tento router UZ neprepina goal na "rear_deep" subgoal. Jediny
// aktivny zasah, ktory router este robi, je tvrdy escape, ked sa robot
// ocitne priamo v high-cost / lethal oblasti lokalnej costmapy (napr.
// ho nieco zatlaci dnu).
//
// Escape policy:
//   - Pozrieme sa, ako daleko siaha lethal pas pozdlz osi robota dopredu
//     a dozadu (`measureAxisClearance` skenuje costmapu krok po kroku).
//   - Default je IST DOPREDU. Robot sa vyberie dozadu iba ked je dozadu
//     pas vyrazne kratsi nez dopredu (rozdiel >= forward_bias_distance_).
//     Priklad pri default 0.5 m bias: 30 cm vpredu vs 10 cm vzadu -> ide
//     dopredu (rozdiel 0.20 < 0.5). 1.0 m vpredu vs 10 cm vzadu -> ide
//     dozadu (rozdiel 0.90 >= 0.5).
//   - Subgoal ulozime v aktualnej orientacii robota (yaw), aby TEB
//     nemusel rotovat v cell s lethal cost; pri spatnom escape robot
//     reverzuje.
// V bezneho stave iba transparentne posiela final goal dalej, aby
// move_base a via-points mohli odviest svoju robotu.
class SafeZoneGoalRouter
{
public:
  SafeZoneGoalRouter()
    : private_nh_("~"),
      tf_buffer_(ros::Duration(10.0)),
      tf_listener_(tf_buffer_),
      has_latest_robot_pose_(false),
      has_latest_costmap_(false),
      has_final_goal_(false),
      has_active_target_(false),
      has_locked_subgoal_(false)
  {
    private_nh_.param<std::string>("input_goal_topic", input_goal_topic_, "/move_base_simple/goal");
    private_nh_.param<std::string>("output_goal_topic", output_goal_topic_, "/managed_move_base_simple/goal");
    private_nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
    private_nh_.param<std::string>("local_costmap_topic", local_costmap_topic_, "/move_base/local_costmap/costmap");
    private_nh_.param<std::string>("output_frame", output_frame_, "odom");

    private_nh_.param("subgoal_reached_distance", subgoal_reached_distance_, 0.25);
    private_nh_.param("goal_reached_distance", goal_reached_distance_, 0.35);
    private_nh_.param("costmap_block_threshold", costmap_block_threshold_, 85);
    private_nh_.param("costmap_check_step", costmap_check_step_, 0.08);
    private_nh_.param("forward_escape_distance", forward_escape_distance_, 1.0);
    private_nh_.param("escape_axis_step", escape_axis_step_, 0.05);
    private_nh_.param("escape_axis_max_distance", escape_axis_max_distance_, 1.5);
    private_nh_.param("forward_bias_distance", forward_bias_distance_, 0.5);
    private_nh_.param("timer_rate", timer_rate_, 10.0);
    private_nh_.param("publish_debug_marker", publish_debug_marker_, true);

    goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(output_goal_topic_, 1, true);
    marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>("debug_marker", 4, true);

    goal_sub_ = nh_.subscribe(input_goal_topic_, 1, &SafeZoneGoalRouter::goalCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 1, &SafeZoneGoalRouter::odomCallback, this);
    costmap_sub_ = nh_.subscribe(local_costmap_topic_, 1, &SafeZoneGoalRouter::costmapCallback, this);

    timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(timer_rate_, 1.0)),
                             &SafeZoneGoalRouter::onTimer, this);
  }

private:
  static double distance2d(const Point2D& a, const Point2D& b)
  {
    return std::hypot(b.x - a.x, b.y - a.y);
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
    active_mode_ = "idle";
    has_active_target_ = false;
    ROS_INFO("SafeZoneGoalRouter: novy final goal prijaty");
    routeGoal(true);
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

  // Skenuje costmapu pozdlz lubovolnej osi (dx, dy) a vrati vzdialenost,
  // pri ktorej hodnota klesne pod lethal threshold. Ak nikdy neopustime
  // lethal pas v ramci escape_axis_max_distance_, vraciame plne max
  // (worst-case clearance). Vystup z mriezky lokalnej costmapy beriem
  // ako "uz mimo zname cierne pole" a teda escape je dosiahnuty.
  double measureAxisClearance(const Point2D& start, double dir_x, double dir_y) const
  {
    if (!has_latest_costmap_)
      return escape_axis_max_distance_;

    const double step = std::max(escape_axis_step_, 1e-3);
    const int max_steps = std::max(1, static_cast<int>(std::ceil(escape_axis_max_distance_ / step)));
    for (int i = 1; i <= max_steps; ++i)
    {
      const double dist = static_cast<double>(i) * step;
      const Point2D probe{start.x + dir_x * dist, start.y + dir_y * dist};
      int value = 0;
      if (!costmapValueAt(probe, &value))
        return dist;
      if (value < costmap_block_threshold_)
        return dist;
    }
    return escape_axis_max_distance_;
  }

  // Vybera escape subgoal na zaklade axialneho clearance. Forward bias:
  // dopredu defaultne; dozadu len ked je tam o forward_bias_distance_ m
  // kratsi pas lethal cost. Subgoal je umiestneny tak, aby presiahol
  // koniec lethal pasu (clear + 0.30 m), s minimom forward_escape_distance_,
  // a orientacia subgoalu zostava aktualny yaw robota.
  struct EscapeChoice
  {
    Point2D xy;
    double yaw;
    bool go_backward;
    double forward_clear;
    double backward_clear;
  };

  EscapeChoice chooseEscape(const geometry_msgs::PoseStamped& pose) const
  {
    tf2::Quaternion q;
    tf2::fromMsg(pose.pose.orientation, q);
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const Point2D start{pose.pose.position.x, pose.pose.position.y};

    const double fwd = measureAxisClearance(start, cy, sy);
    const double bwd = measureAxisClearance(start, -cy, -sy);

    // Strict <: pri remize prefer forward.
    const bool back = (bwd + forward_bias_distance_) < fwd;
    const double dx = back ? -cy : cy;
    const double dy = back ? -sy : sy;
    const double clear = back ? bwd : fwd;
    const double drive = std::max(forward_escape_distance_, clear + 0.30);

    EscapeChoice out;
    out.xy = Point2D{start.x + dx * drive, start.y + dy * drive};
    out.yaw = yaw;
    out.go_backward = back;
    out.forward_clear = fwd;
    out.backward_clear = bwd;
    return out;
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
    if (mode == "escape")
    {
      marker.color.r = 0.98;
      marker.color.g = 0.75;
      marker.color.b = 0.08;
      marker.color.a = 0.95;
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

    // Bez odometrie nedokazeme robit ziadne inteligentne rozhodnutia,
    // takze goal len posleme dalej.
    if (!has_latest_robot_pose_)
    {
      if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
        publishTarget(final_goal_, "final");
      return;
    }

    const Point2D robot_xy = poseToPoint(latest_robot_pose_);
    const Point2D goal_xy = poseToPoint(final_goal_);
    int robot_cost = 0;
    const bool robot_in_blocked_cost =
      costmapValueAt(robot_xy, &robot_cost) && robot_cost >= costmap_block_threshold_;

    if (distance2d(robot_xy, goal_xy) <= goal_reached_distance_)
    {
      has_locked_subgoal_ = false;
      active_mode_ = "idle";
      clearActiveTarget();
      return;
    }

    // Drzime escape subgoal, ktory sme uz vyrobili.
    if (has_locked_subgoal_)
    {
      const Point2D subgoal_xy = poseToPoint(locked_subgoal_);
      const bool reached_escape = distance2d(robot_xy, subgoal_xy) <= subgoal_reached_distance_;
      const bool final_path_clear = !lineBlockedInCostmap(robot_xy, goal_xy);
      if (reached_escape || (!robot_in_blocked_cost && final_path_clear))
      {
        has_locked_subgoal_ = false;
        ROS_INFO("SafeZoneGoalRouter: robot opustil high-cost oblast, vraciam final goal");
        if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
          publishTarget(final_goal_, "final");
        return;
      }

      if (force_publish || !has_active_target_ || active_mode_ != "escape" || poseChanged(active_target_, locked_subgoal_))
        publishTarget(locked_subgoal_, "escape");
      return;
    }

    // Robot uviazol v lethal/high-cost cele: vyber smer (forward s biasom,
    // backward len pri vyrazne kratsom pase) a vystup z pasu pozdlz
    // tej osi. Subgoal drzi current yaw -> pri spatnom escape robot
    // reverzuje bez rotacie.
    if (robot_in_blocked_cost)
    {
      const EscapeChoice choice = chooseEscape(latest_robot_pose_);
      locked_subgoal_ = makePose(choice.xy, final_goal_, choice.yaw, true);
      has_locked_subgoal_ = true;
      ROS_INFO("SafeZoneGoalRouter: high-cost (%d) -> %s escape (fwd_clear=%.2f bwd_clear=%.2f)",
               robot_cost,
               choice.go_backward ? "BACKWARD" : "FORWARD",
               choice.forward_clear, choice.backward_clear);

      if (force_publish || !has_active_target_ || active_mode_ != "escape" || poseChanged(active_target_, locked_subgoal_))
        publishTarget(locked_subgoal_, "escape");
      return;
    }

    // Default: transparentny passthrough. Obchadzanie dynamickej prekazky
    // riesi PredictedSweepRiskLayer cez soft via-points pre TEB.
    if (force_publish || !has_active_target_ || active_mode_ != "final" || poseChanged(active_target_, final_goal_))
      publishTarget(final_goal_, "final");
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
  ros::Subscriber odom_sub_;
  ros::Subscriber costmap_sub_;
  ros::Timer timer_;

  std::string input_goal_topic_;
  std::string output_goal_topic_;
  std::string odom_topic_;
  std::string local_costmap_topic_;
  std::string output_frame_;
  std::string active_mode_;

  bool publish_debug_marker_;
  bool has_latest_robot_pose_;
  bool has_latest_costmap_;
  bool has_final_goal_;
  bool has_active_target_;
  bool has_locked_subgoal_;

  double subgoal_reached_distance_;
  double goal_reached_distance_;
  double costmap_check_step_;
  double forward_escape_distance_;
  double escape_axis_step_;
  double escape_axis_max_distance_;
  double forward_bias_distance_;
  double timer_rate_;

  int costmap_block_threshold_;

  nav_msgs::OccupancyGrid latest_costmap_;
  geometry_msgs::PoseStamped latest_robot_pose_;
  geometry_msgs::PoseStamped final_goal_;
  geometry_msgs::PoseStamped active_target_;
  geometry_msgs::PoseStamped locked_subgoal_;
};

}  // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "safe_zone_goal_router");
  SafeZoneGoalRouter router;
  ros::spin();
  return 0;
}
