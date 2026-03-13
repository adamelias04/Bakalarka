#include <ros/ros.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <string>
#include <vector>

class DynamicObstaclePredictor
{
public:
    DynamicObstaclePredictor()
    {
        ros::NodeHandle pnh("~");

        pnh.param<std::string>("model_name", model_name_, "center_box");
        pnh.param<double>("prediction_time", prediction_time_, 1.0);
        pnh.param<double>("dt", dt_, 0.2);
        pnh.param<std::string>("frame_id", frame_id_, "map");

        model_states_sub_ = nh_.subscribe("/gazebo/model_states", 10, &DynamicObstaclePredictor::modelStatesCallback, this);
        obstacle_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/dynamic_obstacle_pose", 1);
        predicted_path_pub_ = nh_.advertise<nav_msgs::Path>("/predicted_obstacle_path", 1);
        predicted_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/predicted_obstacle_markers", 1);

        has_previous_pose_ = false;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber model_states_sub_;
    ros::Publisher obstacle_pose_pub_;
    ros::Publisher predicted_path_pub_;
    ros::Publisher predicted_marker_pub_;

    std::string model_name_;
    std::string frame_id_;
    double prediction_time_;
    double dt_;

    bool has_previous_pose_;
    geometry_msgs::Pose previous_pose_;
    ros::Time previous_time_;

    void modelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr& msg)
    {
        int model_index = -1;

        for (std::size_t i = 0; i < msg->name.size(); ++i)
        {
            if (msg->name[i] == model_name_)
            {
                model_index = static_cast<int>(i);
                break;
            }
        }

        if (model_index < 0)
        {
            ROS_WARN_THROTTLE(2.0, "Model '%s' not found in /gazebo/model_states", model_name_.c_str());
            return;
        }

        ros::Time current_time = ros::Time::now();
        const geometry_msgs::Pose& current_pose = msg->pose[model_index];

        geometry_msgs::PoseStamped obstacle_pose_msg;
        obstacle_pose_msg.header.stamp = current_time;
        obstacle_pose_msg.header.frame_id = frame_id_;
        obstacle_pose_msg.pose = current_pose;
        obstacle_pose_pub_.publish(obstacle_pose_msg);

        if (!has_previous_pose_)
        {
            previous_pose_ = current_pose;
            previous_time_ = current_time;
            has_previous_pose_ = true;
            return;
        }

        double delta_t = (current_time - previous_time_).toSec();
        if (delta_t <= 0.0)
        {
            return;
        }

        double vx = (current_pose.position.x - previous_pose_.position.x) / delta_t;
        double vy = (current_pose.position.y - previous_pose_.position.y) / delta_t;
        double vz = (current_pose.position.z - previous_pose_.position.z) / delta_t;

        nav_msgs::Path predicted_path_msg;
        predicted_path_msg.header.stamp = current_time;
        predicted_path_msg.header.frame_id = frame_id_;

        visualization_msgs::Marker marker;
        marker.header.stamp = current_time;
        marker.header.frame_id = frame_id_;
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

        for (double t = 0.0; t <= prediction_time_; t += dt_)
        {
            geometry_msgs::PoseStamped pred_pose;
            pred_pose.header.stamp = current_time;
            pred_pose.header.frame_id = frame_id_;

            pred_pose.pose = current_pose;
            pred_pose.pose.position.x = current_pose.position.x + vx * t;
            pred_pose.pose.position.y = current_pose.position.y + vy * t;
            pred_pose.pose.position.z = current_pose.position.z + vz * t;

            predicted_path_msg.poses.push_back(pred_pose);

            geometry_msgs::Point p;
            p.x = pred_pose.pose.position.x;
            p.y = pred_pose.pose.position.y;
            p.z = pred_pose.pose.position.z;
            marker.points.push_back(p);
        }

        predicted_path_pub_.publish(predicted_path_msg);
        predicted_marker_pub_.publish(marker);

        previous_pose_ = current_pose;
        previous_time_ = current_time;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "dynamic_obstacle_predictor");
    DynamicObstaclePredictor predictor;
    ros::spin();
    return 0;
}