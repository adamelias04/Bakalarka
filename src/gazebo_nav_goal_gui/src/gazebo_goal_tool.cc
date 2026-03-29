#include "gazebo_goal_tool.hh"

#include <cmath>
#include <iostream>

#include <QHBoxLayout>
#include <QToolTip>

#include <gazebo/gui/GuiEvents.hh>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/gui/MouseEventHandler.hh>

namespace gazebo
{
GZ_REGISTER_GUI_PLUGIN(GazeboGoalTool)

GazeboGoalTool::GazeboGoalTool() : GUIPlugin()
{
  this->setObjectName("gazeboGoalTool");
  this->setAttribute(Qt::WA_TranslucentBackground, true);
  this->setStyleSheet(
      "QWidget#gazeboGoalTool { background: transparent; }"
      "QPushButton {"
      "  background-color: rgba(34, 34, 34, 210);"
      "  color: white;"
      "  border: 1px solid rgba(220, 220, 220, 120);"
      "  border-radius: 6px;"
      "  padding: 6px 12px;"
      "  font-weight: bold;"
      "}"
      "QPushButton:hover {"
      "  background-color: rgba(56, 56, 56, 225);"
      "}"
      "QPushButton:checked {"
      "  background-color: rgba(210, 105, 30, 230);"
      "  border: 1px solid rgba(255, 220, 180, 220);"
      "}");

  this->toggleButton_ = new QPushButton("2D Nav Goal");
  this->toggleButton_->setCheckable(true);
  this->toggleButton_->setChecked(false);
  this->toggleButton_->setFixedSize(118, 32);
  this->toggleButton_->setToolTip("Activate 2D Nav Goal and click on the map.");

  auto *mainLayout = new QHBoxLayout();
  mainLayout->addWidget(this->toggleButton_);
  mainLayout->setContentsMargins(4, 4, 4, 4);

  this->setLayout(mainLayout);
  this->move(this->posX_, this->posY_);
  this->resize(126, 40);

  QObject::connect(
      this->toggleButton_,
      &QPushButton::toggled,
      [this](bool checked)
      {
        this->OnToggle(checked);
      });
}

GazeboGoalTool::~GazeboGoalTool()
{
  if (gui::MouseEventHandler::Instance())
  {
    gui::MouseEventHandler::Instance()->RemovePressFilter(this->mouseFilterName_);
    gui::MouseEventHandler::Instance()->RemoveReleaseFilter(this->mouseFilterName_);
  }
  this->preRenderConn_.reset();
}

void GazeboGoalTool::Load(sdf::ElementPtr _sdf)
{
  if (_sdf)
  {
    if (_sdf->HasElement("frame_id"))
      this->frameId_ = _sdf->Get<std::string>("frame_id");

    if (_sdf->HasElement("ground_z"))
      this->groundZ_ = _sdf->Get<double>("ground_z");

    if (_sdf->HasElement("x"))
      this->posX_ = _sdf->Get<int>("x");

    if (_sdf->HasElement("y"))
      this->posY_ = _sdf->Get<int>("y");
  }

  this->move(this->posX_, this->posY_);

  if (!ros::isInitialized())
  {
    std::cerr << "[GazeboGoalTool] ROS is not initialized. Run Gazebo through gazebo_ros." << std::endl;
    return;
  }

  this->nh_ = std::make_unique<ros::NodeHandle>("gazebo_goal_tool");
  this->goalPub_ =
      this->nh_->advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, false);

  this->preRenderConn_ = event::Events::ConnectPreRender(
      std::bind(&GazeboGoalTool::OnPreRender, this));

  gui::MouseEventHandler::Instance()->AddPressFilter(
      this->mouseFilterName_,
      std::bind(&GazeboGoalTool::OnMousePress, this, std::placeholders::_1));

  gui::MouseEventHandler::Instance()->AddReleaseFilter(
      this->mouseFilterName_,
      std::bind(&GazeboGoalTool::OnMouseRelease, this, std::placeholders::_1));

  std::cout << "[GazeboGoalTool] Loaded. Publishing to /move_base_simple/goal with frame_id='"
            << this->frameId_ << "'" << std::endl;
}

void GazeboGoalTool::OnToggle(bool _checked)
{
  this->toolEnabled_ = _checked;
  this->dragActive_ = false;

  if (_checked)
  {
    this->toggleButton_->setText("Goal: ON");
    this->toggleButton_->setToolTip(
        "2D Nav Goal active. Click or click-drag on the map to publish a goal.");
  }
  else
  {
    this->toggleButton_->setText("2D Nav Goal");
    this->toggleButton_->setToolTip("Activate 2D Nav Goal and click on the map.");
  }
}

void GazeboGoalTool::OnPreRender()
{
  if (this->cameraReady_)
    return;

  this->userCamera_ = gui::get_active_camera();
  if (this->userCamera_)
  {
    this->cameraReady_ = true;
    std::cout << "[GazeboGoalTool] Active camera acquired." << std::endl;
  }
}

bool GazeboGoalTool::ScreenToGround(
    const common::MouseEvent &_event,
    ignition::math::Vector3d &_worldPoint) const
{
  if (!this->userCamera_)
    return false;

  ignition::math::Planed groundPlane(ignition::math::Vector3d(0, 0, 1), this->groundZ_);
  if (!this->userCamera_->WorldPointOnPlane(
          _event.Pos().X(), _event.Pos().Y(), groundPlane, _worldPoint))
  {
    return false;
  }

  if (!std::isfinite(_worldPoint.X()) ||
      !std::isfinite(_worldPoint.Y()) ||
      !std::isfinite(_worldPoint.Z()))
  {
    return false;
  }

  return true;
}

bool GazeboGoalTool::OnMousePress(const common::MouseEvent &_event)
{
  if (!this->toolEnabled_ || !this->cameraReady_)
    return false;

  if (_event.Button() != common::MouseEvent::LEFT)
    return false;

  ignition::math::Vector3d pt;
  if (!this->ScreenToGround(_event, pt))
    return false;

  this->dragStart_ = pt;
  this->dragActive_ = true;
  return true;
}

bool GazeboGoalTool::OnMouseRelease(const common::MouseEvent &_event)
{
  if (!this->toolEnabled_ || !this->cameraReady_ || !this->dragActive_)
    return false;

  if (_event.Button() != common::MouseEvent::LEFT)
    return false;

  ignition::math::Vector3d pt;
  if (!this->ScreenToGround(_event, pt))
  {
    this->dragActive_ = false;
    return false;
  }

  const ignition::math::Vector2i delta = _event.Pos() - _event.PressPos();
  if (std::abs(delta.X()) < 3 && std::abs(delta.Y()) < 3)
    pt = this->dragStart_;

  this->PublishGoal(this->dragStart_, pt);
  this->dragActive_ = false;
  this->toggleButton_->setChecked(false);
  return true;
}

void GazeboGoalTool::PublishGoal(
    const ignition::math::Vector3d &_start,
    const ignition::math::Vector3d &_end)
{
  const double dx = _end.X() - _start.X();
  const double dy = _end.Y() - _start.Y();

  double yaw = 0.0;
  const double len = std::hypot(dx, dy);
  if (len > 1e-3)
    yaw = std::atan2(dy, dx);

  geometry_msgs::PoseStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = this->frameId_;

  msg.pose.position.x = _start.X();
  msg.pose.position.y = _start.Y();
  msg.pose.position.z = 0.0;

  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = std::sin(yaw * 0.5);
  msg.pose.orientation.w = std::cos(yaw * 0.5);

  this->goalPub_.publish(msg);

  std::cout << "[GazeboGoalTool] Goal published: x=" << msg.pose.position.x
            << ", y=" << msg.pose.position.y
            << ", yaw=" << yaw << std::endl;
}
}  // namespace gazebo
