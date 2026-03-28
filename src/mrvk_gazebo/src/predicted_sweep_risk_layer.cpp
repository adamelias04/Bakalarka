#include <mrvk_gazebo/predicted_sweep_risk_layer.h>

#include <pluginlib/class_list_macros.h>
#include <costmap_2d/cost_values.h>
#include <visualization_msgs/Marker.h>
#include <algorithm>
#include <cmath>

namespace mrvk_gazebo
{

PredictedSweepRiskLayer::PredictedSweepRiskLayer()
    : has_path_(false),
      has_pose_(false),
      visualization_active_(false)
{
}

void PredictedSweepRiskLayer::onInitialize()
{
    ros::NodeHandle nh("~/" + name_);

    current_ = true;

    nh.param<std::string>("path_topic", path_topic_, std::string("/predicted_obstacle_path"));
    nh.param<std::string>("pose_topic", pose_topic_, std::string("/detected_obstacle_pose"));
    nh.param<std::string>("visualization_topic", visualization_topic_, std::string("/predicted_sweep_safe_zone"));

    nh.param("corridor_half_width", corridor_half_width_, 0.45);
    nh.param("side_extra_width", side_extra_width_, 0.30);
    nh.param("obstacle_radius", obstacle_radius_, 0.20);
    nh.param("max_path_length", max_path_length_, 1.50);
    nh.param("velocity_epsilon", velocity_epsilon_, 0.03);
    nh.param("pose_staleness_tolerance", pose_staleness_tolerance_, 0.35);
    nh.param("pose_path_deviation_tolerance", pose_path_deviation_tolerance_, 0.45);
    nh.param("visualization_z", visualization_z_, 0.08);
    nh.param("visualization_height", visualization_height_, 0.03);
    nh.param("visualization_clearance", visualization_clearance_, 0.55);
    nh.param("rear_zone_length_scale", rear_zone_length_scale_, 1.5);
    nh.param("front_zone_length_scale", front_zone_length_scale_, 1.0);

    nh.param("sweep_cost", sweep_cost_, 220);
    nh.param("side_cost", side_cost_, 150);
    nh.param("body_cost", body_cost_, 252);

    nh.param("use_pose_as_start", use_pose_as_start_, true);
    nh.param("publish_visualization", publish_visualization_, true);
    nh.param("require_fresh_observation", require_fresh_observation_, true);

    path_sub_ = nh.subscribe(path_topic_, 1, &PredictedSweepRiskLayer::pathCallback, this);
    pose_sub_ = nh.subscribe(pose_topic_, 1, &PredictedSweepRiskLayer::poseCallback, this);
    marker_pub_ = nh.advertise<visualization_msgs::Marker>(visualization_topic_, 8, true);

    ROS_INFO("[%s] PredictedSweepRiskLayer initialized", name_.c_str());
}

void PredictedSweepRiskLayer::reset()
{
    clearVisualization();
    deactivate();
    activate();
}

void PredictedSweepRiskLayer::pathCallback(const nav_msgs::Path::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    latest_path_ = *msg;
    has_path_ = !latest_path_.poses.empty();
    publishVisualization();
}

void PredictedSweepRiskLayer::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    latest_pose_ = *msg;
    has_pose_ = true;
    publishVisualization();
}

double PredictedSweepRiskLayer::clamp01(double v) const
{
    return std::max(0.0, std::min(1.0, v));
}

double PredictedSweepRiskLayer::norm2d(double x, double y) const
{
    return std::sqrt(x * x + y * y);
}

double PredictedSweepRiskLayer::distance2d(double x0, double y0, double x1, double y1) const
{
    return norm2d(x1 - x0, y1 - y0);
}

bool PredictedSweepRiskLayer::hasUsableGeometry() const
{
    if (!has_path_ || latest_path_.poses.empty())
        return false;

    if (require_fresh_observation_ && !hasFreshObservation())
        return false;

    const Point2D start = selectStartPoint();

    for (const auto& pose_stamped : latest_path_.poses)
    {
        const geometry_msgs::Point& point = pose_stamped.pose.position;
        if (distance2d(start.x, start.y, point.x, point.y) >= velocity_epsilon_)
            return true;
    }

    return false;
}

bool PredictedSweepRiskLayer::hasFreshObservation() const
{
    if (!has_pose_)
        return false;

    const ros::Time path_stamp = latest_path_.header.stamp;
    const ros::Time pose_stamp = latest_pose_.header.stamp;

    if (!path_stamp.isZero() && !pose_stamp.isZero())
    {
        const double dt = std::fabs((path_stamp - pose_stamp).toSec());
        if (dt > pose_staleness_tolerance_)
            return false;
    }

    const double deviation = distance2d(latest_path_.poses.front().pose.position.x,
                                        latest_path_.poses.front().pose.position.y,
                                        latest_pose_.pose.position.x,
                                        latest_pose_.pose.position.y);
    if (deviation > pose_path_deviation_tolerance_)
        return false;

    return true;
}

PredictedSweepRiskLayer::Point2D PredictedSweepRiskLayer::selectStartPoint() const
{
    Point2D start;
    start.x = latest_path_.poses.front().pose.position.x;
    start.y = latest_path_.poses.front().pose.position.y;

    if (!use_pose_as_start_ || !has_pose_)
        return start;

    if (!hasFreshObservation())
        return start;

    start.x = latest_pose_.pose.position.x;
    start.y = latest_pose_.pose.position.y;
    return start;
}

bool PredictedSweepRiskLayer::buildSweepGeometry(SweepGeometry& geometry) const
{
    if (!hasUsableGeometry())
        return false;

    geometry.start = selectStartPoint();

    geometry.end = geometry.start;
    double accumulated = 0.0;

    for (std::size_t i = 1; i < latest_path_.poses.size(); ++i)
    {
        Point2D prev;
        prev.x = latest_path_.poses[i - 1].pose.position.x;
        prev.y = latest_path_.poses[i - 1].pose.position.y;

        Point2D cur;
        cur.x = latest_path_.poses[i].pose.position.x;
        cur.y = latest_path_.poses[i].pose.position.y;

        const double seg = distance2d(prev.x, prev.y, cur.x, cur.y);

        if (accumulated + seg >= max_path_length_)
        {
            const double remain = std::max(0.0, max_path_length_ - accumulated);
            const double ratio = (seg > 1e-6) ? (remain / seg) : 0.0;
            geometry.end.x = prev.x + ratio * (cur.x - prev.x);
            geometry.end.y = prev.y + ratio * (cur.y - prev.y);
            accumulated = max_path_length_;
            break;
        }

        geometry.end = cur;
        accumulated += seg;
    }

    geometry.dir.x = geometry.end.x - geometry.start.x;
    geometry.dir.y = geometry.end.y - geometry.start.y;
    geometry.length = norm2d(geometry.dir.x, geometry.dir.y);

    if (geometry.length < velocity_epsilon_)
        return false;

    geometry.dir.x /= geometry.length;
    geometry.dir.y /= geometry.length;
    geometry.normal.x = -geometry.dir.y;
    geometry.normal.y =  geometry.dir.x;

    return true;
}

PredictedSweepRiskLayer::Point2D PredictedSweepRiskLayer::offsetPoint(const Point2D& point,
                                                                      const Point2D& dir,
                                                                      double distance) const
{
    Point2D shifted;
    shifted.x = point.x + dir.x * distance;
    shifted.y = point.y + dir.y * distance;
    return shifted;
}

PredictedSweepRiskLayer::ZoneGeometry PredictedSweepRiskLayer::computeRearZone(const SweepGeometry& geometry) const
{
    ZoneGeometry zone;
    zone.near_point = offsetPoint(geometry.start, geometry.dir, -(obstacle_radius_ + visualization_clearance_));
    zone.dir.x = -geometry.dir.x;
    zone.dir.y = -geometry.dir.y;
    zone.length = std::max(geometry.length * rear_zone_length_scale_, 1e-3);
    return zone;
}

PredictedSweepRiskLayer::ZoneGeometry PredictedSweepRiskLayer::computeFrontZone(const SweepGeometry& geometry) const
{
    ZoneGeometry zone;
    zone.near_point = offsetPoint(geometry.end, geometry.dir, visualization_clearance_);
    zone.dir = geometry.dir;
    zone.length = std::max(geometry.length * front_zone_length_scale_, 1e-3);
    return zone;
}

PredictedSweepRiskLayer::Point2D PredictedSweepRiskLayer::computeZoneCenter(const ZoneGeometry& zone) const
{
    return offsetPoint(zone.near_point, zone.dir, 0.5 * zone.length);
}

unsigned char PredictedSweepRiskLayer::computeZoneCostAt(double wx, double wy,
                                                         const SweepGeometry& geometry,
                                                         const ZoneGeometry& zone) const
{
    const double dx = wx - zone.near_point.x;
    const double dy = wy - zone.near_point.y;

    const double longitudinal = dx * zone.dir.x + dy * zone.dir.y;
    const double lateral = std::fabs(dx * geometry.normal.x + dy * geometry.normal.y);

    if (longitudinal < 0.0 || longitudinal > zone.length)
        return costmap_2d::FREE_SPACE;

    if (lateral <= corridor_half_width_)
    {
        const double long_gain = 1.0 - (longitudinal / std::max(zone.length, 1e-6));
        const double lat_gain  = 1.0 - (lateral / std::max(corridor_half_width_, 1e-6));
        const double gain = clamp01(long_gain) * (0.6 + 0.4 * clamp01(lat_gain));
        const int cost = static_cast<int>(std::round(sweep_cost_ * gain));
        return static_cast<unsigned char>(std::min(252, cost));
    }

    if (lateral <= corridor_half_width_ + side_extra_width_)
    {
        const double long_gain = 1.0 - (longitudinal / std::max(zone.length, 1e-6));
        const double lat_norm = (lateral - corridor_half_width_) / std::max(side_extra_width_, 1e-6);
        const double lat_gain = 1.0 - lat_norm;
        const double gain = clamp01(long_gain) * clamp01(lat_gain);
        const int cost = static_cast<int>(std::round(side_cost_ * gain));
        return static_cast<unsigned char>(std::min(252, cost));
    }

    return costmap_2d::FREE_SPACE;
}

void PredictedSweepRiskLayer::updateBounds(double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
                                           double* min_x, double* min_y, double* max_x, double* max_y)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!enabled_)
        return;

    SweepGeometry geometry;
    if (!buildSweepGeometry(geometry))
        return;

    const double pad = std::max(obstacle_radius_, corridor_half_width_ + side_extra_width_) + 0.2;
    const ZoneGeometry rear_zone = computeRearZone(geometry);
    const ZoneGeometry front_zone = computeFrontZone(geometry);
    const Point2D rear_far = offsetPoint(rear_zone.near_point, rear_zone.dir, rear_zone.length);
    const Point2D front_far = offsetPoint(front_zone.near_point, front_zone.dir, front_zone.length);

    *min_x = std::min(*min_x, geometry.start.x - pad);
    *min_y = std::min(*min_y, geometry.start.y - pad);
    *max_x = std::max(*max_x, geometry.start.x + pad);
    *max_y = std::max(*max_y, geometry.start.y + pad);

    *min_x = std::min(*min_x, rear_zone.near_point.x - pad);
    *min_y = std::min(*min_y, rear_zone.near_point.y - pad);
    *max_x = std::max(*max_x, rear_zone.near_point.x + pad);
    *max_y = std::max(*max_y, rear_zone.near_point.y + pad);

    *min_x = std::min(*min_x, rear_far.x - pad);
    *min_y = std::min(*min_y, rear_far.y - pad);
    *max_x = std::max(*max_x, rear_far.x + pad);
    *max_y = std::max(*max_y, rear_far.y + pad);

    *min_x = std::min(*min_x, front_zone.near_point.x - pad);
    *min_y = std::min(*min_y, front_zone.near_point.y - pad);
    *max_x = std::max(*max_x, front_zone.near_point.x + pad);
    *max_y = std::max(*max_y, front_zone.near_point.y + pad);

    *min_x = std::min(*min_x, front_far.x - pad);
    *min_y = std::min(*min_y, front_far.y - pad);
    *max_x = std::max(*max_x, front_far.x + pad);
    *max_y = std::max(*max_y, front_far.y + pad);
}

unsigned char PredictedSweepRiskLayer::computeCostAt(double wx, double wy) const
{
    SweepGeometry geometry;
    if (!buildSweepGeometry(geometry))
        return costmap_2d::FREE_SPACE;

    const double dist_to_start = distance2d(wx, wy, geometry.start.x, geometry.start.y);
    if (dist_to_start <= obstacle_radius_)
        return static_cast<unsigned char>(std::min(252, body_cost_));

    const unsigned char rear_cost = computeZoneCostAt(wx, wy, geometry, computeRearZone(geometry));
    const unsigned char front_cost = computeZoneCostAt(wx, wy, geometry, computeFrontZone(geometry));
    return std::max(rear_cost, front_cost);
}

void PredictedSweepRiskLayer::publishVisualization()
{
    if (!publish_visualization_ || !marker_pub_)
        return;

    SweepGeometry geometry;
    if (!buildSweepGeometry(geometry))
    {
        clearVisualization();
        return;
    }

    const ros::Time stamp = latest_path_.header.stamp.isZero() ? ros::Time::now() : latest_path_.header.stamp;
    const double yaw = std::atan2(geometry.dir.y, geometry.dir.x);
    const double half_yaw = 0.5 * yaw;
    const double qz = std::sin(half_yaw);
    const double qw = std::cos(half_yaw);
    const double outer_half_width = corridor_half_width_ + side_extra_width_;
    const ZoneGeometry rear_zone = computeRearZone(geometry);
    const ZoneGeometry front_zone = computeFrontZone(geometry);
    const Point2D rear_center = computeZoneCenter(rear_zone);
    const Point2D front_center = computeZoneCenter(front_zone);

    visualization_msgs::Marker rear_outer_strip;
    rear_outer_strip.header.stamp = stamp;
    rear_outer_strip.header.frame_id = latest_path_.header.frame_id;
    rear_outer_strip.ns = "predicted_sweep_safe_zone";
    rear_outer_strip.id = 0;
    rear_outer_strip.type = visualization_msgs::Marker::CUBE;
    rear_outer_strip.action = visualization_msgs::Marker::ADD;
    rear_outer_strip.pose.position.x = rear_center.x;
    rear_outer_strip.pose.position.y = rear_center.y;
    rear_outer_strip.pose.position.z = visualization_z_;
    rear_outer_strip.pose.orientation.z = qz;
    rear_outer_strip.pose.orientation.w = qw;
    rear_outer_strip.scale.x = std::max(rear_zone.length, 1e-3);
    rear_outer_strip.scale.y = std::max(2.0 * outer_half_width, 1e-3);
    rear_outer_strip.scale.z = std::max(visualization_height_, 1e-3);
    rear_outer_strip.color.r = 1.0;
    rear_outer_strip.color.g = 0.95;
    rear_outer_strip.color.b = 0.10;
    rear_outer_strip.color.a = 0.18;

    visualization_msgs::Marker rear_inner_strip = rear_outer_strip;
    rear_inner_strip.id = 1;
    rear_inner_strip.scale.y = std::max(2.0 * corridor_half_width_, 1e-3);
    rear_inner_strip.color.r = 1.0;
    rear_inner_strip.color.g = 0.35;
    rear_inner_strip.color.b = 0.10;
    rear_inner_strip.color.a = 0.32;

    visualization_msgs::Marker front_outer_strip = rear_outer_strip;
    front_outer_strip.id = 2;
    front_outer_strip.pose.position.x = front_center.x;
    front_outer_strip.pose.position.y = front_center.y;
    front_outer_strip.scale.x = std::max(front_zone.length, 1e-3);
    front_outer_strip.color.r = 0.95;
    front_outer_strip.color.g = 0.90;
    front_outer_strip.color.b = 0.20;
    front_outer_strip.color.a = 0.18;

    visualization_msgs::Marker front_inner_strip = rear_inner_strip;
    front_inner_strip.id = 3;
    front_inner_strip.pose.position.x = front_center.x;
    front_inner_strip.pose.position.y = front_center.y;
    front_inner_strip.scale.x = std::max(front_zone.length, 1e-3);
    front_inner_strip.color.r = 1.0;
    front_inner_strip.color.g = 0.50;
    front_inner_strip.color.b = 0.12;
    front_inner_strip.color.a = 0.32;

    visualization_msgs::Marker obstacle_body = rear_outer_strip;
    obstacle_body.id = 4;
    obstacle_body.type = visualization_msgs::Marker::CYLINDER;
    obstacle_body.pose.position.x = geometry.start.x;
    obstacle_body.pose.position.y = geometry.start.y;
    obstacle_body.pose.orientation.z = 0.0;
    obstacle_body.pose.orientation.w = 1.0;
    obstacle_body.scale.x = std::max(2.0 * obstacle_radius_, 1e-3);
    obstacle_body.scale.y = std::max(2.0 * obstacle_radius_, 1e-3);
    obstacle_body.scale.z = std::max(visualization_height_ * 1.2, 1e-3);
    obstacle_body.color.r = 0.85;
    obstacle_body.color.g = 0.10;
    obstacle_body.color.b = 0.10;
    obstacle_body.color.a = 0.55;

    marker_pub_.publish(rear_outer_strip);
    marker_pub_.publish(rear_inner_strip);
    marker_pub_.publish(front_outer_strip);
    marker_pub_.publish(front_inner_strip);
    marker_pub_.publish(obstacle_body);
    visualization_active_ = true;
}

void PredictedSweepRiskLayer::clearVisualization()
{
    if (!publish_visualization_ || !marker_pub_ || !visualization_active_)
        return;

    const std::string frame_id = !latest_path_.header.frame_id.empty()
        ? latest_path_.header.frame_id
        : (!latest_pose_.header.frame_id.empty() ? latest_pose_.header.frame_id : layered_costmap_->getGlobalFrameID());
    const ros::Time stamp = ros::Time::now();

    for (int id = 0; id < 5; ++id)
    {
        visualization_msgs::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = frame_id;
        marker.ns = "predicted_sweep_safe_zone";
        marker.id = id;
        marker.action = visualization_msgs::Marker::DELETE;
        marker_pub_.publish(marker);
    }

    visualization_active_ = false;
}

void PredictedSweepRiskLayer::updateCosts(costmap_2d::Costmap2D& master_grid,
                                          int min_i, int min_j, int max_i, int max_j)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!enabled_ || !hasUsableGeometry())
    {
        clearVisualization();
        return;
    }

    publishVisualization();

    for (int j = min_j; j < max_j; ++j)
    {
        for (int i = min_i; i < max_i; ++i)
        {
            double wx, wy;
            master_grid.mapToWorld(i, j, wx, wy);

            unsigned char risk_cost = computeCostAt(wx, wy);
            if (risk_cost == costmap_2d::FREE_SPACE)
                continue;

            unsigned char old_cost = master_grid.getCost(i, j);

            if (old_cost == costmap_2d::NO_INFORMATION)
                master_grid.setCost(i, j, risk_cost);
            else
                master_grid.setCost(i, j, std::max(old_cost, risk_cost));
        }
    }
}

}  // namespace mrvk_gazebo

PLUGINLIB_EXPORT_CLASS(mrvk_gazebo::PredictedSweepRiskLayer, costmap_2d::Layer)
