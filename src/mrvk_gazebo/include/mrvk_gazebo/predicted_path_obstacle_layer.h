#ifndef MRVK_GAZEBO_PREDICTED_PATH_OBSTACLE_LAYER_H_
#define MRVK_GAZEBO_PREDICTED_PATH_OBSTACLE_LAYER_H_

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <nav_msgs/Path.h>

#include <mutex>
#include <string>
#include <vector>

namespace mrvk_gazebo
{

// Zapisuje LETHAL bunky pozdlz /predicted_obstacle_path priamo do master
// gridu v updateCosts. Bez perzistencie — kazdy costmap update cyklus sa
// prepocita nanovo z aktualneho path, takze stare predikcie zmiznu akonahle
// publisher posle novy (alebo prestal publikovat — path ide stale).
class PredictedPathObstacleLayer : public costmap_2d::Layer
{
public:
    PredictedPathObstacleLayer();

    virtual void onInitialize();
    virtual void updateBounds(double robot_x, double robot_y, double robot_yaw,
                              double* min_x, double* min_y,
                              double* max_x, double* max_y);
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

    void pathCallback(const nav_msgs::Path::ConstPtr& msg);

    std::vector<Point2D> buildTruncatedPath() const;
    bool hasFreshPath(const ros::Time& now) const;

    double pointToSegmentDistSq(double px, double py,
                                double x0, double y0,
                                double x1, double y1) const;

    void expandBounds(double* min_x, double* min_y,
                      double* max_x, double* max_y,
                      double px, double py, double pad) const;

    std::mutex mutex_;

    ros::Subscriber path_sub_;

    nav_msgs::Path latest_path_;
    bool has_path_;

    double robot_x_;
    double robot_y_;
    bool has_robot_pose_;

    std::string path_topic_;
    double obstacle_radius_;
    double max_path_length_;
    double robot_exclusion_radius_;
    double path_staleness_tolerance_;
    int cost_value_;
};

}  // namespace mrvk_gazebo

#endif
