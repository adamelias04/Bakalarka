#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import copy

class SimpleFilter(object):
    def __init__(self, odom_filtered_pub):
        self.odom_filtered_pub_ = odom_filtered_pub

    def IMU_callback(self, data):
        self.IMUData = data

    def odometry_callback(self, data):
        filteredData = Odometry()
        filteredData = copy.deepcopy(data)
        filteredData.orientation.x = (data.orientation.x + self.IMUData.orientation.x)/2
        filteredData.orientation.y = (data.orientation.y + self.IMUData.orientation.y)/2
        filteredData.orientation.z = (data.orientation.z + self.IMUData.orientation.z)/2
        filteredData.orientation.w = (data.orientation.w + self.IMUData.orientation.w)/2

        self.odom_filtered_pub.publish(filteredData)

def main():
    rospy.init_node('imu_quat_to_eul_node')
    odom_filtered_pub = rospy.Publisher('/odom_filtered', Odometry, queue_size = 10)    

    simpleFilter = SimpleFilter(odom_filtered_pub)
    rospy.Subscriber("/imu_data", Imu, simpleFilter.IMU_callback)
    rospy.Subscriber("/shoddy/odom", Odometry, simpleFilter.odometry_callback)
    rospy.spin()

if __name__ == '__main__':
    main()