#!/usr/bin/env python

import math
import time

import os.path
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from imu_quat_to_eul.msg import EulRot
from tf.transformations import euler_from_quaternion

class Transformer(object):
    def __init__(self, pub_imu, pub_odom):
        self.odomFullPath = os.path.join(os.path.expanduser('~'), 'Documents', "odomRot.txt")
        self.IMUFullPath = os.path.join(os.path.expanduser('~'), 'Documents', "imuRot.txt")
        open(self.odomFullPath, 'w').close()
        open(self.IMUFullPath, 'w').close()

        self.pub_imu = pub_imu
        self.pub_odom = pub_odom           

    def writeToFile(self, secs, nsecs, x, y, z, fullpath):     
        f = open(fullpath, "a")        

        f.write("{} {} {:.4f} {:.4f} {:.4f}  \n".format(secs, nsecs, x, y, z))
        f.close()        

    def transform_callback(self, data):        
        euler = euler_from_quaternion([data.orientation.x, 
                                       data.orientation.y, 
                                       data.orientation.z, 
                                       data.orientation.w])               
        msg = EulRot()
        msg.time = data.header.stamp
        msg.x = euler[0]
        msg.y = euler[1]
        msg.z = euler[2]        
        self.pub_imu.publish(msg)
        self.writeToFile(data.header.stamp.secs,
                         data.header.stamp.nsecs, 
                         euler[0], euler[1], euler[2], self.IMUFullPath)        

    def odometry_callback(self, data):        
        euler = euler_from_quaternion([data.pose.pose.orientation.x, 
                                       data.pose.pose.orientation.y, 
                                       data.pose.pose.orientation.z, 
                                       data.pose.pose.orientation.w])                   
        msg = EulRot()
        msg.time = data.header.stamp
        msg.x = euler[0]
        msg.y = euler[1]
        msg.z = euler[2]        
        self.pub_odom.publish(msg)        
        self.writeToFile(data.header.stamp.secs,
                         data.header.stamp.nsecs, 
                         euler[0], euler[1], euler[2], self.odomFullPath)


def main():
    ref = (0, 0)

    rospy.init_node('imu_quat_to_eul_node')
    pub_imu = rospy.Publisher('/imu_quat_to_eul_top', EulRot, queue_size = 10)    
    pub_odom = rospy.Publisher('/odom_quat_to_eul_top', EulRot, queue_size = 10)

    transformer = Transformer(pub_imu, pub_odom)    
    rospy.Subscriber("/imu_data", Imu, transformer.transform_callback)
    rospy.Subscriber("/shoddy/odom", Odometry, transformer.odometry_callback)
    rospy.spin()

if __name__ == '__main__':
    main()