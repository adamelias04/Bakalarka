#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point32.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>

struct ScanPoint
{
    double x_laser;
    double y_laser;
    double range;
    double angle;
    int index;
};

struct Segment
{
    std::vector<ScanPoint> points;
    double mean_range;
    double mean_angle;
    double width;
};

struct Point2D
{
    double x;
    double y;
};

class DynamicObstaclePredictor
{
public:
    DynamicObstaclePredictor()
        : tf_listener_(tf_buffer_),
          has_track_(false)
    {
        ros::NodeHandle pnh("~");

        pnh.param<std::string>("scan_topic", scan_topic_, "/scan");
        pnh.param<std::string>("output_frame", output_frame_, "odom");

        pnh.param<double>("prediction_time", prediction_time_, 1.0);
        pnh.param<double>("prediction_dt", prediction_dt_, 0.1);

        pnh.param<double>("alpha_gain", alpha_gain_, 0.65);
        pnh.param<double>("beta_gain", beta_gain_, 0.20);

        pnh.param<double>("max_speed", max_speed_, 2.0);
        pnh.param<double>("track_timeout", track_timeout_, 0.8);
        pnh.param<double>("lost_track_prediction_time", lost_track_prediction_time_, 1.0);

        pnh.param<double>("point_spacing", point_spacing_, 0.05);
        pnh.param<double>("corridor_radius", corridor_radius_, 0.18);
        pnh.param<double>("z_height", z_height_, 0.10);

        pnh.param<double>("scan_range_min_use", scan_range_min_use_, 0.10);
        pnh.param<double>("scan_range_max_use", scan_range_max_use_, 8.00);

        pnh.param<double>("segment_jump_distance", segment_jump_distance_, 0.30);
        pnh.param<int>("segment_min_points", segment_min_points_, 2);
        pnh.param<int>("segment_max_points", segment_max_points_, 120);

        pnh.param<double>("segment_min_width", segment_min_width_, 0.03);
        pnh.param<double>("segment_max_width", segment_max_width_, 2.00);

        pnh.param<bool>("use_angle_roi", use_angle_roi_, false);
        pnh.param<double>("angle_min_roi", angle_min_roi_, -1.57);
        pnh.param<double>("angle_max_roi", angle_max_roi_,  1.57);

        pnh.param<bool>("use_range_roi", use_range_roi_, false);
        pnh.param<double>("range_min_roi", range_min_roi_, 0.20);
        pnh.param<double>("range_max_roi", range_max_roi_, 8.00);

        pnh.param<double>("initial_expected_angle", initial_expected_angle_, 0.0);
        pnh.param<double>("initial_expected_range", initial_expected_range_, 2.0);

        pnh.param<double>("max_association_distance", max_association_distance_, 1.5);

        pnh.param<bool>("publish_debug_segment", publish_debug_segment_, true);

        scan_sub_ = nh_.subscribe(scan_topic_, 1, &DynamicObstaclePredictor::scanCallback, this);

        detected_pose_pub_    = nh_.advertise<geometry_msgs::PoseStamped>("/detected_obstacle_pose", 1);
        predicted_path_pub_   = nh_.advertise<nav_msgs::Path>("/predicted_obstacle_path", 1);
        predicted_cloud_pub_  = nh_.advertise<sensor_msgs::PointCloud>("/predicted_obstacle_cloud", 1);
        predicted_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/predicted_obstacle_markers", 1);
        debug_segment_pub_    = nh_.advertise<visualization_msgs::Marker>("/predicted_obstacle_debug_segment", 1);
        velocity_marker_pub_  = nh_.advertise<visualization_msgs::Marker>("/predicted_obstacle_velocity_marker", 1);

        resetTrack();

        ROS_INFO("dynamic_obstacle_predictor started");
        ROS_INFO("scan_topic: %s", scan_topic_.c_str());
        ROS_INFO("output_frame: %s", output_frame_.c_str());
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher detected_pose_pub_;
    ros::Publisher predicted_path_pub_;
    ros::Publisher predicted_cloud_pub_;
    ros::Publisher predicted_marker_pub_;
    ros::Publisher debug_segment_pub_;
    ros::Publisher velocity_marker_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::string scan_topic_;
    std::string output_frame_;

    double prediction_time_;
    double prediction_dt_;

    double alpha_gain_;
    double beta_gain_;
    double max_speed_;
    double track_timeout_;
    double lost_track_prediction_time_;

    double point_spacing_;
    double corridor_radius_;
    double z_height_;

    double scan_range_min_use_;
    double scan_range_max_use_;

    double segment_jump_distance_;
    int segment_min_points_;
    int segment_max_points_;
    double segment_min_width_;
    double segment_max_width_;

    bool use_angle_roi_;
    double angle_min_roi_;
    double angle_max_roi_;

    bool use_range_roi_;
    double range_min_roi_;
    double range_max_roi_;

    double initial_expected_angle_;
    double initial_expected_range_;
    double max_association_distance_;

    bool publish_debug_segment_;

    bool has_track_;
    ros::Time last_track_stamp_;
    Point2D tracked_position_;
    double vx_;
    double vy_;

    void resetTrack()
    {
        has_track_ = false;
        last_track_stamp_ = ros::Time(0);
        tracked_position_.x = 0.0;
        tracked_position_.y = 0.0;
        vx_ = 0.0;
        vy_ = 0.0;
    }

    double clamp(double v, double lo, double hi) const
    {
        return std::max(lo, std::min(hi, v));
    }

    bool validRange(float r) const
    {
        return std::isfinite(r) && r >= scan_range_min_use_ && r <= scan_range_max_use_;
    }

    bool insideAngleROI(double angle) const
    {
        if (!use_angle_roi_)
            return true;
        return angle >= angle_min_roi_ && angle <= angle_max_roi_;
    }

    bool insideRangeROI(double range) const
    {
        if (!use_range_roi_)
            return true;
        return range >= range_min_roi_ && range <= range_max_roi_;
    }

    double distLaser(const ScanPoint& a, const ScanPoint& b) const
    {
        const double dx = a.x_laser - b.x_laser;
        const double dy = a.y_laser - b.y_laser;
        return std::sqrt(dx * dx + dy * dy);
    }

    double distWorld(const Point2D& a, const Point2D& b) const
    {
        const double dx = a.x - b.x;
        const double dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    std::vector<ScanPoint> extractValidPoints(const sensor_msgs::LaserScan::ConstPtr& scan_msg)
    {
        std::vector<ScanPoint> pts;
        pts.reserve(scan_msg->ranges.size());

        double angle = scan_msg->angle_min;

        for (std::size_t i = 0; i < scan_msg->ranges.size(); ++i, angle += scan_msg->angle_increment)
        {
            const float r = scan_msg->ranges[i];

            if (!validRange(r))
                continue;

            if (!insideAngleROI(angle))
                continue;

            if (!insideRangeROI(r))
                continue;

            ScanPoint p;
            p.range = static_cast<double>(r);
            p.angle = angle;
            p.x_laser = p.range * std::cos(angle);
            p.y_laser = p.range * std::sin(angle);
            p.index = static_cast<int>(i);

            pts.push_back(p);
        }

        return pts;
    }

    void finalizeSegment(const std::vector<ScanPoint>& raw_segment, std::vector<Segment>& out_segments)
    {
        if (raw_segment.size() < static_cast<std::size_t>(segment_min_points_))
            return;

        if (raw_segment.size() > static_cast<std::size_t>(segment_max_points_))
            return;

        Segment s;
        s.points = raw_segment;

        double sum_r = 0.0;
        double sum_a = 0.0;
        for (const auto& p : raw_segment)
        {
            sum_r += p.range;
            sum_a += p.angle;
        }

        s.mean_range = sum_r / static_cast<double>(raw_segment.size());
        s.mean_angle = sum_a / static_cast<double>(raw_segment.size());

        const ScanPoint& first = raw_segment.front();
        const ScanPoint& last  = raw_segment.back();
        s.width = distLaser(first, last);

        if (s.width < segment_min_width_ || s.width > segment_max_width_)
            return;

        out_segments.push_back(s);
    }

    std::vector<Segment> buildSegments(const std::vector<ScanPoint>& pts)
    {
        std::vector<Segment> segments;
        if (pts.empty())
            return segments;

        std::vector<ScanPoint> current;
        current.push_back(pts.front());

        for (std::size_t i = 1; i < pts.size(); ++i)
        {
            if (distLaser(pts[i - 1], pts[i]) <= segment_jump_distance_)
            {
                current.push_back(pts[i]);
            }
            else
            {
                finalizeSegment(current, segments);
                current.clear();
                current.push_back(pts[i]);
            }
        }

        finalizeSegment(current, segments);
        return segments;
    }

    Point2D segmentMidpointLaser(const Segment& s) const
    {
        const ScanPoint& a = s.points.front();
        const ScanPoint& b = s.points.back();

        Point2D p;
        p.x = 0.5 * (a.x_laser + b.x_laser);
        p.y = 0.5 * (a.y_laser + b.y_laser);
        return p;
    }

    bool laserPointToWorld(const Point2D& p_laser,
                           const std::string& laser_frame,
                           const ros::Time& stamp,
                           Point2D& p_world)
    {
        geometry_msgs::PointStamped in_pt;
        geometry_msgs::PointStamped out_pt;

        in_pt.header.stamp = stamp;
        in_pt.header.frame_id = laser_frame;
        in_pt.point.x = p_laser.x;
        in_pt.point.y = p_laser.y;
        in_pt.point.z = 0.0;

        try
        {
            tf_buffer_.transform(in_pt, out_pt, output_frame_, ros::Duration(0.1));
        }
        catch (const tf2::TransformException& ex)
        {
            ROS_WARN_THROTTLE(1.0, "Point transform failed: %s", ex.what());
            return false;
        }

        p_world.x = out_pt.point.x;
        p_world.y = out_pt.point.y;
        return true;
    }

    bool selectSegment(const std::vector<Segment>& segments,
                       const sensor_msgs::LaserScan::ConstPtr& scan_msg,
                       Segment& selected,
                       Point2D& selected_world)
    {
        if (segments.empty())
            return false;

        double best_score = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        Point2D best_world;

        for (std::size_t i = 0; i < segments.size(); ++i)
        {
            Point2D p_laser = segmentMidpointLaser(segments[i]);
            Point2D p_world;

            if (!laserPointToWorld(p_laser, scan_msg->header.frame_id, scan_msg->header.stamp, p_world))
                continue;

            double score = 0.0;

            if (has_track_)
            {
                const double d = distWorld(p_world, tracked_position_);
                if (d > max_association_distance_)
                    continue;

                score = d;
            }
            else
            {
                const double da = segments[i].mean_angle - initial_expected_angle_;
                const double dr = segments[i].mean_range - initial_expected_range_;
                score = std::sqrt(da * da + dr * dr);
            }

            if (score < best_score)
            {
                best_score = score;
                best_idx = static_cast<int>(i);
                best_world = p_world;
            }
        }

        if (best_idx < 0)
            return false;

        selected = segments[best_idx];
        selected_world = best_world;
        return true;
    }

    void publishDetectedPose(const Point2D& p, const ros::Time& stamp)
    {
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = output_frame_;
        pose.pose.orientation.w = 1.0;
        pose.pose.position.x = p.x;
        pose.pose.position.y = p.y;
        pose.pose.position.z = z_height_;
        detected_pose_pub_.publish(pose);
    }

    void addDisc(sensor_msgs::PointCloud& cloud, double cx, double cy, double cz)
    {
        for (double ox = -corridor_radius_; ox <= corridor_radius_; ox += point_spacing_)
        {
            for (double oy = -corridor_radius_; oy <= corridor_radius_; oy += point_spacing_)
            {
                if (ox * ox + oy * oy <= corridor_radius_ * corridor_radius_)
                {
                    geometry_msgs::Point32 p;
                    p.x = cx + ox;
                    p.y = cy + oy;
                    p.z = cz;
                    cloud.points.push_back(p);
                }
            }
        }
    }

    void addDenseSegment(sensor_msgs::PointCloud& cloud,
                         double x0, double y0, double z0,
                         double x1, double y1, double z1)
    {
        const double dx = x1 - x0;
        const double dy = y1 - y0;
        const double dz = z1 - z0;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        const int steps = std::max(1, static_cast<int>(std::ceil(dist / point_spacing_)));

        for (int i = 0; i <= steps; ++i)
        {
            const double r = static_cast<double>(i) / static_cast<double>(steps);
            const double px = x0 + r * dx;
            const double py = y0 + r * dy;
            const double pz = z0 + r * dz;
            addDisc(cloud, px, py, pz);
        }
    }

    void publishDebugSegment(const Segment& seg,
                             const std::string& laser_frame,
                             const ros::Time& stamp)
    {
        if (!publish_debug_segment_)
            return;

        visualization_msgs::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = output_frame_;
        marker.ns = "predicted_obstacle_debug_segment";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::POINTS;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.06;
        marker.scale.y = 0.06;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        for (const auto& sp : seg.points)
        {
            Point2D p_laser;
            p_laser.x = sp.x_laser;
            p_laser.y = sp.y_laser;

            Point2D p_world;
            if (!laserPointToWorld(p_laser, laser_frame, stamp, p_world))
                continue;

            geometry_msgs::Point p;
            p.x = p_world.x;
            p.y = p_world.y;
            p.z = z_height_;
            marker.points.push_back(p);
        }

        debug_segment_pub_.publish(marker);
    }

    void publishVelocityMarker(const ros::Time& stamp)
    {
        if (!has_track_)
            return;

        visualization_msgs::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = output_frame_;
        marker.ns = "predicted_obstacle_velocity";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        marker.scale.y = 0.10;
        marker.scale.z = 0.10;
        marker.color.r = 0.0;
        marker.color.g = 0.4;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        geometry_msgs::Point p0;
        geometry_msgs::Point p1;

        p0.x = tracked_position_.x;
        p0.y = tracked_position_.y;
        p0.z = z_height_;

        p1.x = tracked_position_.x + 0.5 * vx_;
        p1.y = tracked_position_.y + 0.5 * vy_;
        p1.z = z_height_;

        marker.points.push_back(p0);
        marker.points.push_back(p1);

        velocity_marker_pub_.publish(marker);
    }

    void publishPrediction(const ros::Time& stamp, double horizon)
    {
        if (!has_track_)
            return;

        nav_msgs::Path path;
        path.header.stamp = stamp;
        path.header.frame_id = output_frame_;

        visualization_msgs::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = output_frame_;
        marker.ns = "predicted_obstacle";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        sensor_msgs::PointCloud cloud;
        cloud.header.stamp = stamp;
        cloud.header.frame_id = output_frame_;

        double last_x = tracked_position_.x;
        double last_y = tracked_position_.y;
        double last_z = z_height_;

        for (double t = 0.0; t <= horizon + 1e-6; t += prediction_dt_)
        {
            const double px = tracked_position_.x + vx_ * t;
            const double py = tracked_position_.y + vy_ * t;
            const double pz = z_height_;

            geometry_msgs::PoseStamped pose;
            pose.header.stamp = stamp;
            pose.header.frame_id = output_frame_;
            pose.pose.orientation.w = 1.0;
            pose.pose.position.x = px;
            pose.pose.position.y = py;
            pose.pose.position.z = pz;
            path.poses.push_back(pose);

            geometry_msgs::Point p;
            p.x = px;
            p.y = py;
            p.z = pz;
            marker.points.push_back(p);

            addDenseSegment(cloud, last_x, last_y, last_z, px, py, pz);

            last_x = px;
            last_y = py;
            last_z = pz;
        }

        predicted_path_pub_.publish(path);
        predicted_marker_pub_.publish(marker);
        predicted_cloud_pub_.publish(cloud);
        publishVelocityMarker(stamp);
    }

    void initializeTrack(const Point2D& p, const ros::Time& stamp)
    {
        tracked_position_ = p;
        vx_ = 0.0;
        vy_ = 0.0;
        last_track_stamp_ = stamp;
        has_track_ = true;
    }

    void updateTrack(const Point2D& measurement, double dt)
    {
        Point2D predicted;
        predicted.x = tracked_position_.x + vx_ * dt;
        predicted.y = tracked_position_.y + vy_ * dt;

        const double rx = measurement.x - predicted.x;
        const double ry = measurement.y - predicted.y;

        tracked_position_.x = predicted.x + alpha_gain_ * rx;
        tracked_position_.y = predicted.y + alpha_gain_ * ry;

        if (dt > 1e-4)
        {
            vx_ += (beta_gain_ / dt) * rx;
            vy_ += (beta_gain_ / dt) * ry;
        }

        vx_ = clamp(vx_, -max_speed_, max_speed_);
        vy_ = clamp(vy_, -max_speed_, max_speed_);
    }

    void handleMissingDetection(const ros::Time& stamp, const std::string& reason)
    {
        if (!has_track_)
        {
            ROS_WARN_THROTTLE(1.0, "%s", reason.c_str());
            return;
        }

        const double dt = (stamp - last_track_stamp_).toSec();

        if (dt < 0.0 || dt > track_timeout_)
        {
            ROS_WARN_THROTTLE(1.0, "%s, track expired", reason.c_str());
            resetTrack();
            return;
        }

        tracked_position_.x += vx_ * dt;
        tracked_position_.y += vy_ * dt;
        last_track_stamp_ = stamp;

        ROS_WARN_THROTTLE(1.0, "%s, using predicted continuation", reason.c_str());
        publishPrediction(stamp, lost_track_prediction_time_);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_msg)
    {
        const std::vector<ScanPoint> valid_pts = extractValidPoints(scan_msg);

        if (valid_pts.empty())
        {
            handleMissingDetection(scan_msg->header.stamp, "No valid scan points after ROI filtering");
            return;
        }

        const std::vector<Segment> segments = buildSegments(valid_pts);

        if (segments.empty())
        {
            handleMissingDetection(scan_msg->header.stamp, "No valid scan segments");
            return;
        }

        Segment selected_segment;
        Point2D measurement_world;

        if (!selectSegment(segments, scan_msg, selected_segment, measurement_world))
        {
            handleMissingDetection(scan_msg->header.stamp, "No segment selected");
            return;
        }

        publishDebugSegment(selected_segment, scan_msg->header.frame_id, scan_msg->header.stamp);
        publishDetectedPose(measurement_world, scan_msg->header.stamp);

        if (!has_track_)
        {
            initializeTrack(measurement_world, scan_msg->header.stamp);
            publishPrediction(scan_msg->header.stamp, prediction_time_);
            return;
        }

        double dt = (scan_msg->header.stamp - last_track_stamp_).toSec();

        if (dt < 0.0 || dt > track_timeout_)
        {
            initializeTrack(measurement_world, scan_msg->header.stamp);
            publishPrediction(scan_msg->header.stamp, prediction_time_);
            return;
        }

        if (dt < 1e-4)
        {
            publishPrediction(scan_msg->header.stamp, prediction_time_);
            return;
        }

        updateTrack(measurement_world, dt);
        last_track_stamp_ = scan_msg->header.stamp;

        publishPrediction(scan_msg->header.stamp, prediction_time_);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "dynamic_obstacle_predictor");
    DynamicObstaclePredictor node;
    ros::spin();
    return 0;
}