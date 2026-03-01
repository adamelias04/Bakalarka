#ifndef DETECTED_PATH_LAYER_H_
#define DETECTED_PATH_LAYER_H_

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <dynamic_reconfigure/server.h>
#include <path_detection/DetectedPath.h>
#include <vector>
#include <math.h>

#define PI        3.1415926536

namespace costmap_2d {
    class DetectedPathLayer : public costmap_2d::Layer {
        public:
            /// Constructors-Destructors:
            DetectedPathLayer();
            
            /// Methods:
            virtual void onInitialize();
            virtual void updateBounds(double origin_x, double origin_y, double origin_yaw, 
                                      double *min_x, double *min_y, double *max_x, double *max_y);
            virtual void updateCosts(costmap_2d::Costmap2D &master_grid, int min_i, int min_j, int max_i, int max_j);
    
        private:
            /// Methods:
            void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
            void PathDetectionCallback(const path_detection::DetectedPath::ConstPtr &msg);
            void frame_to_costmap();
            // void copyArray(int *arr, const int *source);
            
            /// Properties:
            dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
            ros::Subscriber sub;

            double robot_x, robot_y, robot_yaw;
            double min_x, min_y, max_x, max_y;
            
            unsigned int FRAME_HEIGHT;
            unsigned int FRAME_WIDTH;
            std::vector<unsigned char> frame;

            std::vector<double> mark_x;
            std::vector<double> mark_y;
            const unsigned int PENALTY = 252;

            // Camera parameters
            const double CAMERA_FOCAL_LENGTH = 474.4268835;     // Focal length
            const double CAMERA_X = 0.1946454825;               // Camera x-shift from the center of the robot
            const double CAMERA_HEIGHT = 0.5706493713;          // Camera height
            const double CAMERA_PITCH = PI/8;                   // Camera pitch
            const double PHI = PI - CAMERA_PITCH;
            const double COS_CAMERA_PITCH = cos(CAMERA_PITCH);
            const double SIN_CAMERA_PITCH = sin(CAMERA_PITCH);
            const unsigned int FRAME_STARTING_IDX = 280;        // The first index of the field of vision of the robot
            // const unsigned int FRAME_STARTING_IDX = ceil(CAMERA_FOCAL_LENGTH*tan(CAMERA_PITCH - PI));   // The index of the first positive measurement
            const unsigned int FRAME_ENDING_IDX = 550;          // The last index of the field of vision of the robot
    };
}

#endif