#include <mrvk_gazebo/predicted_path_obstacle_layer.h>

#include <costmap_2d/cost_values.h>
#include <pluginlib/class_list_macros.h>

#include <algorithm>
#include <cmath>
#include <limits>

PLUGINLIB_EXPORT_CLASS(mrvk_gazebo::PredictedPathObstacleLayer, costmap_2d::Layer)

namespace mrvk_gazebo
{

PredictedPathObstacleLayer::PredictedPathObstacleLayer()
    : has_path_(false),
      robot_x_(0.0),
      robot_y_(0.0),
      has_robot_pose_(false),
      obstacle_radius_(0.25),
      max_path_length_(3.0),
      robot_exclusion_radius_(0.55),
      path_staleness_tolerance_(0.5),
      cost_value_(static_cast<int>(costmap_2d::LETHAL_OBSTACLE))
{
}

void PredictedPathObstacleLayer::onInitialize()
{
    ros::NodeHandle nh("~/" + name_);
    current_ = true;
    enabled_ = true;

    nh.param("enabled", enabled_, true);
    nh.param<std::string>("path_topic", path_topic_, "/predicted_obstacle_path");
    nh.param("obstacle_radius", obstacle_radius_, 0.25);
    nh.param("max_path_length", max_path_length_, 3.0);
    nh.param("robot_exclusion_radius", robot_exclusion_radius_, 0.55);
    nh.param("path_staleness_tolerance", path_staleness_tolerance_, 0.5);
    nh.param("cost_value", cost_value_, static_cast<int>(costmap_2d::LETHAL_OBSTACLE));

    cost_value_ = std::min(std::max(cost_value_, 1),
                           static_cast<int>(costmap_2d::LETHAL_OBSTACLE));

    ros::NodeHandle g_nh;
    path_sub_ = g_nh.subscribe(path_topic_, 1,
                               &PredictedPathObstacleLayer::pathCallback, this);

    ROS_INFO("PredictedPathObstacleLayer initialised (topic=%s radius=%.2f max_len=%.2f excl=%.2f)",
             path_topic_.c_str(), obstacle_radius_,
             max_path_length_, robot_exclusion_radius_);
}

void PredictedPathObstacleLayer::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    has_path_ = false;
    latest_path_.poses.clear();
}

void PredictedPathObstacleLayer::pathCallback(const nav_msgs::Path::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    latest_path_ = *msg;
    has_path_ = !latest_path_.poses.empty();
}

bool PredictedPathObstacleLayer::hasFreshPath(const ros::Time& now) const
{
    if (!has_path_) return false;
    if (latest_path_.header.stamp.isZero()) return false;
    const double age = (now - latest_path_.header.stamp).toSec();
    return age >= 0.0 && age <= path_staleness_tolerance_;
}

std::vector<PredictedPathObstacleLayer::Point2D>
PredictedPathObstacleLayer::buildTruncatedPath() const
{
    std::vector<Point2D> pts;
    if (!has_path_ || latest_path_.poses.empty()) return pts;

    pts.reserve(latest_path_.poses.size());
    double accumulated = 0.0;
    bool has_prev = false;
    Point2D prev{0.0, 0.0};

    for (const auto& pose : latest_path_.poses)
    {
        Point2D cur{pose.pose.position.x, pose.pose.position.y};

        if (!has_prev)
        {
            pts.push_back(cur);
            prev = cur;
            has_prev = true;
            continue;
        }

        const double dx = cur.x - prev.x;
        const double dy = cur.y - prev.y;
        const double seg = std::hypot(dx, dy);
        if (seg < 1e-9) { prev = cur; continue; }

        if (accumulated + seg > max_path_length_)
        {
            const double remain = std::max(0.0, max_path_length_ - accumulated);
            const double ratio = remain / seg;
            Point2D truncated{prev.x + ratio * dx, prev.y + ratio * dy};
            pts.push_back(truncated);
            accumulated += remain;
            break;
        }

        pts.push_back(cur);
        accumulated += seg;
        prev = cur;
    }

    // Staticka/takmer staticka prekazka: path pose posty su takmer vsetky
    // na rovnakom mieste, co by vyrobilo male lethal jadro. Tej uz rozumie
    // scan_marking v ObstacleLayeri (prekazka je aj v realnom scane), takze
    // duplikaciu preskakujeme — inak by inflation_layer rozfukol mali disc
    // do velkeho blobu a zbytocne by blokoval planovanie.
    if (accumulated < 0.20) pts.clear();

    return pts;
}

void PredictedPathObstacleLayer::updateBounds(double robot_x, double robot_y,
                                               double /*robot_yaw*/,
                                               double* min_x, double* min_y,
                                               double* max_x, double* max_y)
{
    std::lock_guard<std::mutex> lock(mutex_);
    robot_x_ = robot_x;
    robot_y_ = robot_y;
    has_robot_pose_ = true;

    if (!enabled_) return;
    if (!hasFreshPath(ros::Time::now())) return;

    const auto path = buildTruncatedPath();
    if (path.size() < 2) return;

    const double pad = obstacle_radius_ + 0.1;
    for (const auto& p : path)
        expandBounds(min_x, min_y, max_x, max_y, p.x, p.y, pad);
}

void PredictedPathObstacleLayer::updateCosts(costmap_2d::Costmap2D& master_grid,
                                              int min_i, int min_j,
                                              int max_i, int max_j)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    if (!hasFreshPath(ros::Time::now())) return;

    const auto path = buildTruncatedPath();
    if (path.size() < 2) return;

    const double rad2 = obstacle_radius_ * obstacle_radius_;
    const double excl2 = robot_exclusion_radius_ * robot_exclusion_radius_;
    const unsigned char cost = static_cast<unsigned char>(cost_value_);

    for (int j = min_j; j < max_j; ++j)
    {
        for (int i = min_i; i < max_i; ++i)
        {
            double wx = 0.0, wy = 0.0;
            master_grid.mapToWorld(static_cast<unsigned int>(i),
                                    static_cast<unsigned int>(j), wx, wy);

            if (has_robot_pose_)
            {
                const double dxr = wx - robot_x_;
                const double dyr = wy - robot_y_;
                if (dxr * dxr + dyr * dyr < excl2)
                    continue;
            }

            double best = std::numeric_limits<double>::infinity();
            for (std::size_t k = 1; k < path.size(); ++k)
            {
                const double d2 = pointToSegmentDistSq(wx, wy,
                                                       path[k - 1].x, path[k - 1].y,
                                                       path[k].x, path[k].y);
                if (d2 < best) best = d2;
                if (best < 1e-9) break;
            }

            if (best > rad2) continue;

            const unsigned char old = master_grid.getCost(i, j);
            master_grid.setCost(i, j,
                                (old == costmap_2d::NO_INFORMATION)
                                    ? cost
                                    : std::max(old, cost));
        }
    }
}

double PredictedPathObstacleLayer::pointToSegmentDistSq(double px, double py,
                                                         double x0, double y0,
                                                         double x1, double y1) const
{
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double len2 = dx * dx + dy * dy;
    if (len2 < 1e-12)
    {
        const double ddx = px - x0;
        const double ddy = py - y0;
        return ddx * ddx + ddy * ddy;
    }
    double t = ((px - x0) * dx + (py - y0) * dy) / len2;
    t = std::max(0.0, std::min(1.0, t));
    const double cx = x0 + t * dx;
    const double cy = y0 + t * dy;
    const double ex = px - cx;
    const double ey = py - cy;
    return ex * ex + ey * ey;
}

void PredictedPathObstacleLayer::expandBounds(double* min_x, double* min_y,
                                               double* max_x, double* max_y,
                                               double px, double py, double pad) const
{
    *min_x = std::min(*min_x, px - pad);
    *min_y = std::min(*min_y, py - pad);
    *max_x = std::max(*max_x, px + pad);
    *max_y = std::max(*max_y, py + pad);
}

}  // namespace mrvk_gazebo
