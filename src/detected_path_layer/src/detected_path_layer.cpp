
#include <detected_path_layer/detected_path_layer.h>
#include <pluginlib/class_list_macros.h>
#include <iostream>

PLUGINLIB_EXPORT_CLASS (costmap_2d::DetectedPathLayer, costmap_2d::Layer) // Register the plug

using costmap_2d::LETHAL_OBSTACLE;

namespace costmap_2d {
    DetectedPathLayer::DetectedPathLayer() {}

    void DetectedPathLayer::onInitialize() {
        ros::NodeHandle nh("~/" + name_);
        current_ = true;
        dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
        dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = boost::bind(&DetectedPathLayer::reconfigureCB, this, _1, _2);
        dsrv_->setCallback(cb);
        sub = nh.subscribe("/shoddy/detected_path", 10, &DetectedPathLayer::PathDetectionCallback, this);
    }

    void DetectedPathLayer::reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level) {
        enabled_ = config.enabled;
    }

    void DetectedPathLayer::updateBounds(double origin_x, double origin_y, double origin_yaw, 
                                double *min_x, double *min_y, double *max_x, double *max_y) {  
        if (!enabled_) {
            return;
        }

        robot_x = origin_x;
        robot_y = origin_y;
        robot_yaw = origin_yaw;

        ros::spinOnce();

        if (!(mark_x.empty() && mark_y.empty())) {
            *min_x = std::min(*min_x, this->min_x);
            *min_y = std::min(*min_y, this->min_y);
            *max_x = std::max(*max_x, this->max_x);
            *max_y = std::max(*max_y, this->max_y);
        }
    }
    
    void DetectedPathLayer::updateCosts(costmap_2d::Costmap2D &master_grid, int min_i, int min_j, int max_i, int max_j) {
        if (!enabled_) {
            return;
        }

        unsigned int x;
        unsigned int y;

        for (unsigned int i = 0; i < mark_x.size(); i++) {
            if (master_grid.worldToMap(mark_x[i], mark_y[i], x, y)) {
                master_grid.setCost(x, y, LETHAL_OBSTACLE);
            }
        }
    }

    void DetectedPathLayer::PathDetectionCallback(const path_detection::DetectedPath::ConstPtr &msg) {
        FRAME_HEIGHT = msg->height;
        FRAME_WIDTH = msg->width;
        frame = msg->frame;

        frame_to_costmap();
    }

    void DetectedPathLayer::frame_to_costmap() {
        mark_x.clear();
        mark_y.clear();

        double phi = 0.0;
        double r = 0.0;
        double x = 0.0;
        double y = 0.0;

        for (unsigned int h = FRAME_STARTING_IDX; h < FRAME_ENDING_IDX; h++) {
            for (unsigned int w = 0; w < FRAME_WIDTH; w++) {
                if (!frame[h*FRAME_WIDTH + w]) {
                    phi = atan2(h - FRAME_HEIGHT/2.0, CAMERA_FOCAL_LENGTH);
                    
                    r = -CAMERA_HEIGHT*(cos(PHI - phi)/sin(PHI - phi));
                    x = CAMERA_X + r;

                    r = r*COS_CAMERA_PITCH + CAMERA_HEIGHT*SIN_CAMERA_PITCH;
                    y = -r/CAMERA_FOCAL_LENGTH*(w - FRAME_WIDTH/2.0);

                    mark_x.push_back(robot_x + x*cos(robot_yaw) - y*sin(robot_yaw));
                    mark_y.push_back(robot_y + x*sin(robot_yaw) + y*cos(robot_yaw));
                }
            }
        }

        if (!(mark_x.empty() && mark_y.empty())) {
            min_x = *std::min_element(mark_x.begin(), mark_x.end());
            min_y = *std::min_element(mark_y.begin(), mark_y.end());
            max_x = *std::max_element(mark_x.begin(), mark_x.end());
            max_y = *std::max_element(mark_y.begin(), mark_y.end());
        }
    }
}