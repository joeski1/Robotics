#!/usr/bin/env python
# node should react to readings from a laser scanner and stop if it gets too
# near an obstacle
# created by Charlie Street 29/09/16
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import LaserScan

def laser_callback(laser_data):
    #rospy.loginfo('{} || {} || {}'.format(
        #laser_data.angle_min, laser_data.angle_max, laser_data.angle_increment))
    rospy.loginfo('{} || {}'.format(laser_data.ranges[256], len(laser_data.ranges)))

    new_speed = Twist() # an object of twist, has linear and angular velocity fields
    read = laser_data.ranges[256]

    if read < 0.75:
        new_speed.linear.x  = -5.0*read # backwards
    elif read > 1:
        new_speed.linear.x  = 10 # forwards
    else:
        new_speed.angular.x = 0.0 # stop

    pub.publish(new_speed)

if __name__ == '__main__':
    rospy.init_node('obstacle_stopper')

    rospy.Subscriber('base_scan', LaserScan, laser_callback)

    # cmd_vel channel to send velocity commands/updates
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)

    # run until shutdown
    rospy.spin()
