#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

# must remove the mapping from des_vel to base_controller/command in teleop_joy.launch
# rostopic echo raw_joy

slow_rotation = 0.3
slow_translation = 0.5

def joystick_callback(joy):
    global publishing
    if joy.buttons[4] == 1 or publishing: # Only publish if left trigger is down
        publishing = True

        axes = list(joy.axes) # "tuple does not support item assignment"
        print('before {}'.format(axes))

        if joy.buttons[5] != 1: # right trigger pressed: normal speed
            axes[0] = slow_rotation * axes[0] # vel w (rotation)
            axes[1] = slow_translation * axes[1] # vel x (forward backwards)

        print('after {}'.format(axes))
        joy.axes = tuple(axes)
        joy_pub.publish(joy)

        # Ensure one more joy message is sent when the left trigger is released
        if joy.buttons[4] != 1:
            print("Left trigger release joy message sent")
            publishing = False

def joy_cmd_vel_callback(twist):
    global publishing
    global stopped_robot
    if publishing:
        stopped_robot = False
        cmd_pub.publish(twist)
    elif not stopped_robot:
        print("Published stop")
        stopped_robot = True
        cmd_pub.publish(Twist())
    else: # not publishing, stopped
        pass # do nothing


if __name__ == '__main__':
    global publishing
    global stopped_robot
    publishing = False
    stopped_robot = True

    rospy.init_node('slow_joystick')

    joy_pub = rospy.Publisher('joy', Joy, queue_size=1)
    cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('raw_joy', Joy, joystick_callback)
    rospy.Subscriber('joy_cmd_vel', Twist, joy_cmd_vel_callback)
    rospy.spin()
