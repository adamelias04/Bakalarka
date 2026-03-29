#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

#include <gazebo/common/MouseEvent.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/gui/GuiPlugin.hh>
#include <gazebo/rendering/rendering.hh>

#include <QPushButton>

namespace gazebo
{
class GazeboGoalTool : public GUIPlugin
{
public:
  GazeboGoalTool();
  virtual ~GazeboGoalTool();

  void Load(sdf::ElementPtr _sdf) override;

private:
  void OnToggle(bool _checked);
  void OnPreRender();
  bool OnMousePress(const common::MouseEvent &_event);
  bool OnMouseRelease(const common::MouseEvent &_event);

  bool ScreenToGround(
      const common::MouseEvent &_event,
      ignition::math::Vector3d &_worldPoint) const;

  void PublishGoal(
      const ignition::math::Vector3d &_start,
      const ignition::math::Vector3d &_end);

private:
  QPushButton *toggleButton_{nullptr};

  bool toolEnabled_{false};
  bool dragActive_{false};
  bool cameraReady_{false};

  std::string mouseFilterName_{"gazebo_goal_tool_filter"};
  std::string frameId_{"odom"};
  int posX_{18};
  int posY_{42};
  double groundZ_{0.0};

  ignition::math::Vector3d dragStart_;

  rendering::UserCameraPtr userCamera_;
  event::ConnectionPtr preRenderConn_;

  std::unique_ptr<ros::NodeHandle> nh_;
  ros::Publisher goalPub_;
};
}  // namespace gazebo
