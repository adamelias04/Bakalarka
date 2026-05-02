// Posiela navigacne ciele do move_base v alternujucom rezime medzi
// dvoma bodmi (default (3,0) a (-3,0)) v ramci frame "map".
// Po dosiahnuti kazdeho ciela cak nahodny cas (default 0-10s) a posle dalsi.
//
// Parametre (~private):
//   ~num_goals    (int,    10)
//   ~min_pause    (double, 0.0)   spodna hranica nahodnej pauzy [s]
//   ~max_pause    (double, 10.0)  horna hranica nahodnej pauzy [s]
//   ~point_a_x/y  (double, 3.0/0.0)
//   ~point_b_x/y  (double, -3.0/0.0)
//   ~frame_id     (string, "odom")  pozn. cely nav stack tu bezi v odom
//   ~goal_timeout (double, 120.0) max cas na jeden ciel [s]
//   ~seed         (int,    -1)    >=0 zafixuje RNG seed pre opakovatelnost

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <utility>

using MoveBaseClient = actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction>;

namespace {

const char* stateName(actionlib::SimpleClientGoalState::StateEnum s) {
  switch (s) {
    case actionlib::SimpleClientGoalState::PENDING:    return "PENDING";
    case actionlib::SimpleClientGoalState::ACTIVE:     return "ACTIVE";
    case actionlib::SimpleClientGoalState::PREEMPTED:  return "PREEMPTED";
    case actionlib::SimpleClientGoalState::SUCCEEDED:  return "SUCCEEDED";
    case actionlib::SimpleClientGoalState::ABORTED:    return "ABORTED";
    case actionlib::SimpleClientGoalState::REJECTED:   return "REJECTED";
    case actionlib::SimpleClientGoalState::RECALLED:   return "RECALLED";
    case actionlib::SimpleClientGoalState::LOST:       return "LOST";
    default:                                           return "UNKNOWN";
  }
}

move_base_msgs::MoveBaseGoal makeGoal(const std::string& frame_id,
                                      double x, double y, double yaw) {
  move_base_msgs::MoveBaseGoal g;
  g.target_pose.header.frame_id = frame_id;
  g.target_pose.header.stamp    = ros::Time::now();
  g.target_pose.pose.position.x = x;
  g.target_pose.pose.position.y = y;
  g.target_pose.pose.position.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  g.target_pose.pose.orientation = tf2::toMsg(q);
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "goal_pinger");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  int    num_goals    = pnh.param("num_goals",    10);
  double min_pause    = pnh.param("min_pause",    0.0);
  double max_pause    = pnh.param("max_pause",    10.0);
  double goal_timeout = pnh.param("goal_timeout", 120.0);
  std::string frame_id = pnh.param<std::string>("frame_id", std::string("odom"));

  double ax = pnh.param("point_a_x",  3.0);
  double ay = pnh.param("point_a_y",  0.0);
  double bx = pnh.param("point_b_x", -3.0);
  double by = pnh.param("point_b_y",  0.0);

  std::vector<std::pair<double, double>> points = {{ax, ay}, {bx, by}};

  int seed = pnh.param("seed", -1);
  std::mt19937 rng;
  if (seed >= 0) {
    rng.seed(static_cast<uint32_t>(seed));
  } else {
    std::random_device rd;
    rng.seed(rd());
  }
  if (max_pause < min_pause) std::swap(min_pause, max_pause);
  std::uniform_real_distribution<double> pause_dist(min_pause, max_pause);

  MoveBaseClient client("move_base", true);
  ROS_INFO("[goal_pinger] cakam na move_base action server...");
  if (!client.waitForServer(ros::Duration(30.0))) {
    ROS_ERROR("[goal_pinger] move_base action server nedostupny.");
    return 1;
  }
  ROS_INFO("[goal_pinger] pripojene k move_base.");

  for (int i = 0; i < num_goals && ros::ok(); ++i) {
    const auto& cur  = points[i % 2];
    const auto& next = points[(i + 1) % 2];
    // natoc robota do smeru daleho ciela, aby pokracoval plynulo
    double yaw = std::atan2(next.second - cur.second, next.first - cur.first);

    ROS_INFO("[goal_pinger] ciel %d/%d -> (%.2f, %.2f) yaw=%.2f",
             i + 1, num_goals, cur.first, cur.second, yaw);
    client.sendGoal(makeGoal(frame_id, cur.first, cur.second, yaw));

    bool finished = client.waitForResult(ros::Duration(goal_timeout));
    if (!finished) {
      ROS_WARN("[goal_pinger] ciel %d timeout (%.1fs), rusim a pokracujem.",
               i + 1, goal_timeout);
      client.cancelGoal();
    } else {
      auto state = client.getState();
      ROS_INFO("[goal_pinger] ciel %d ukonceny, state=%s",
               i + 1, stateName(state.state_));
    }

    if (!ros::ok()) break;

    if (i < num_goals - 1) {
      double pause = pause_dist(rng);
      ROS_INFO("[goal_pinger] cakam %.2fs pred dalsim cielom", pause);
      ros::Duration(pause).sleep();
    }
  }

  ROS_INFO("[goal_pinger] hotovo, %d cielov dokonceno.", num_goals);
  return 0;
}
