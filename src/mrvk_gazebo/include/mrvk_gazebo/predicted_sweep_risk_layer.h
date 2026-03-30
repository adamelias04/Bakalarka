#ifndef MRVK_GAZEBO_PREDICTED_SWEEP_RISK_LAYER_H_
#define MRVK_GAZEBO_PREDICTED_SWEEP_RISK_LAYER_H_

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <mutex>
#include <string>
#include <vector>

namespace mrvk_gazebo
{

class PredictedSweepRiskLayer : public costmap_2d::Layer
{
public:
    PredictedSweepRiskLayer();

    virtual void onInitialize();
    virtual void updateBounds(double robot_x, double robot_y, double robot_yaw,
                              double* min_x, double* min_y, double* max_x, double* max_y);
    virtual void updateCosts(costmap_2d::Costmap2D& master_grid,
                             int min_i, int min_j, int max_i, int max_j);
    virtual void reset();
    virtual bool isDiscretized() { return true; }

private:
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

    void pathCallback(const nav_msgs::Path::ConstPtr& msg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

    bool hasUsableGeometry() const;
    bool hasFreshObservation() const;
    bool canUseHeldMotion() const;
    bool canUseCachedGeometry() const;
    Point2D selectStartPoint() const;
    bool buildSweepGeometry(SweepGeometry& geometry) const;
    Point2D offsetPoint(const Point2D& point, const Point2D& dir, double distance) const;
    ZoneGeometry computeRearZone(const SweepGeometry& geometry) const;
    ZoneGeometry computeFrontZone(const SweepGeometry& geometry) const;
    Point2D computeZoneCenter(const ZoneGeometry& zone) const;
    ZoneGeometry computeUnsafeMiddleZone(const SweepGeometry& geometry) const;
    ZoneGeometry selectPreferredViaZone(const SweepGeometry& geometry) const;
    void appendZoneViaPoints(const ZoneGeometry& zone, std::vector<Point2D>* points) const;
    std::vector<Point2D> buildViaPoints(const SweepGeometry& geometry) const;
    void stampCircleCost(costmap_2d::Costmap2D& master_grid,
                         double wx, double wy,
                         double radius,
                         unsigned char cost) const;
    void applyPathTubeCosts(costmap_2d::Costmap2D& master_grid) const;
    double pointToSegmentDistance(double px, double py,
                                  double x0, double y0,
                                  double x1, double y1) const;
    unsigned char computePathTubeCostAt(double wx, double wy) const;
    unsigned char computeZoneCostAt(double wx, double wy,
                                    const SweepGeometry& geometry,
                                    const ZoneGeometry& zone) const;
    unsigned char computeMiddleZoneCostAt(double wx, double wy,
                                          const SweepGeometry& geometry,
                                          const ZoneGeometry& zone) const;
    unsigned char computeCostAt(double wx, double wy, const SweepGeometry* geometry = nullptr) const;
    void publishVisualization();
    void clearVisualization();
    void publishViaPoints(const SweepGeometry& geometry);
    void clearViaPoints();
    void publishViaPointMarkers(const nav_msgs::Path& path);
    void clearViaPointMarkers();

    double clamp01(double v) const;
    double norm2d(double x, double y) const;
    double distance2d(double x0, double y0, double x1, double y1) const;

    std::mutex mutex_;

    ros::Subscriber path_sub_;
    ros::Subscriber pose_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher via_points_pub_;
    ros::Publisher via_points_marker_pub_;

    nav_msgs::Path latest_path_;
    geometry_msgs::PoseStamped latest_pose_;

    bool has_path_;
    bool has_pose_;
    bool visualization_active_;
    bool via_points_active_;
    bool via_points_marker_active_;
    mutable bool has_cached_motion_;
    mutable Point2D cached_motion_dir_;
    mutable double cached_motion_length_;
    mutable ros::Time cached_motion_stamp_;
    mutable bool has_cached_geometry_;
    mutable SweepGeometry cached_geometry_;
    mutable ros::Time cached_geometry_stamp_;
    bool has_robot_pose_;
    Point2D last_robot_pose_;
    bool has_cached_via_points_;
    nav_msgs::Path cached_via_points_;
    ros::Time cached_via_points_stamp_;

    std::string path_topic_;
    std::string pose_topic_;
    std::string visualization_topic_;
    std::string via_points_topic_;
    std::string via_points_marker_topic_;

    double corridor_half_width_;
    double side_extra_width_;
    double obstacle_radius_;
    double max_path_length_;
    double velocity_epsilon_;
    double pose_staleness_tolerance_;
    double pose_path_deviation_tolerance_;
    double visualization_z_;
    double visualization_height_;
    double visualization_clearance_;
    double rear_zone_length_scale_;
    double front_zone_length_scale_;
    double predicted_path_half_width_;
    double low_speed_hold_time_;
    double min_fallback_length_;
    double geometry_hold_time_;
    double via_points_hold_time_;

    int sweep_cost_;
    int side_cost_;
    int body_cost_;
    int path_cost_;
    bool use_pose_as_start_;
    bool publish_visualization_;
    bool publish_via_points_;
    bool require_fresh_observation_;
};

}  // namespace mrvk_gazebo

#endif
