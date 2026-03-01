#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <string>

namespace gazebo
{
class OscillateBoxPlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
  {
    this->model_ = _model;

    // defaults
    axis_ = "y";
    amplitude_ = 1.0;
    frequency_ = 0.1;

    if (_sdf && _sdf->HasElement("axis"))
      axis_ = _sdf->Get<std::string>("axis");

    if (_sdf && _sdf->HasElement("amplitude"))
      amplitude_ = _sdf->Get<double>("amplitude");

    if (_sdf && _sdf->HasElement("frequency"))
      frequency_ = _sdf->Get<double>("frequency");

    // remember start pose
    start_pose_ = model_->WorldPose();
    start_time_ = model_->GetWorld()->SimTime();

    // connect update callback
    update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&OscillateBoxPlugin::OnUpdate, this));
  }

private:
  void OnUpdate()
  {
    if (!model_) return;

    const common::Time t = model_->GetWorld()->SimTime() - start_time_;
    const double omega = 2.0 * M_PI * frequency_;
    const double offset = amplitude_ * sin(omega * t.Double());

    ignition::math::Pose3d pose = start_pose_;

    if (axis_ == "x") pose.Pos().X() += offset;
    else if (axis_ == "y") pose.Pos().Y() += offset;
    else if (axis_ == "z") pose.Pos().Z() += offset;

    // move model (kinematic style)
    model_->SetWorldPose(pose);
  }

  physics::ModelPtr model_;
  event::ConnectionPtr update_connection_;

  ignition::math::Pose3d start_pose_;
  common::Time start_time_;

  std::string axis_;
  double amplitude_;
  double frequency_;
};

// IMPORTANT: register as MODEL plugin
GZ_REGISTER_MODEL_PLUGIN(OscillateBoxPlugin)
} // namespace gazebo
