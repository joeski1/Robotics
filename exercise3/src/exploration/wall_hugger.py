#!/usr/bin/env python
import math
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


# Check if there seems to be an object in front of the robot, on the data from the left value to the right value, if any of them are less than minRange then return's false
def freeFront(data, left, right, minRange):
    for i in range(left, right):
        ranger = data[i]
        if ranger < minRange:
            return False

    return True


# Returns true if the robot can move forward the specified distance
# (Takes into account width of robot)
def canMoveForward(data, distance):
    robotWidth = 1.0
    # How many reading it will check (distributed evenly across 180 degrees)
    values = range(1, 30)
    for i in values:
        maxReading = abs((robotWidth/2)/math.cos(math.radians((180/values[-1])*i)))
        if maxReading > distance:
            maxReading = distance
        reading = data[(len(data) / values[-1]) * i]
        #rospy.loginfo("Reading "+str(maxReading) + " ~ index "+str((len(data) / values[-1]) * i))
        if reading < maxReading:
            return reading
    return -1


def findAverage(data, low, high):
    temp = 0;
    count = 0;
    for i in range(low, high):
        if (data[i] != float('NaN') and data[i] != float('Inf')):
            temp = temp + data[i]
            count = count + 1
    if count == 0:
        return 20
    else:
        return temp / count


def findGreater(data, fLow, fHigh, sLow, sHigh):
    firstAverage = findAverage(data, fLow, fHigh)
    secondAverage = findAverage(data, sLow, sHigh)

    rospy.loginfo(
        "Front " + str(firstAverage) + "; Back " + str(secondAverage) + " = " + str(firstAverage - secondAverage))

    return firstAverage - secondAverage


def laser_callback(laser_data):
    global detectedWall

    new_speed = Twist()

    maxSpeed = 0.3# 0.25
    maxRotateSpeed = 0.3# 0.3  # negative value = rotate right
    divider = 1

    # Values of laser indicators that indicate front
    frontLeft = 200
    frontRight = 300
    minRange = 0.75

    # Detected first wall
    maxRangeLeft = 1.5
    desRangeLeftMin = 0.75
    desRangeLeftMax = 0.76

    # Range to indicate back right
    firstLower = 0
    firstHigher = 64
    # Range to indicate front right
    secondLower = 64
    secondHigher = 128

    rospy.loginfo("STUFF " + str(detectedWall))

    collision = canMoveForward(laser_data.ranges, minRange)

    if collision != -1:
        # find new wall by rotating
        if collision < 0.4:
            new_speed.linear.x = 0.0
        else:
            new_speed.linear.x = maxSpeed * 0.5
        new_speed.angular.z = maxRotateSpeed * 2
        rospy.loginfo("Rotating for collision ")
    else:
        new_speed.linear.x = maxSpeed
        averageLeft = findAverage(laser_data.ranges, firstLower, secondHigher)
        # if have a wall to our right
        if averageLeft < maxRangeLeft:
            if detectedWall < 100:
                detectedWall += 5
            if desRangeLeftMin <= averageLeft <= desRangeLeftMax:
                # wall is nicely away from us
                rospy.loginfo("Wall is nicely away from us")
                new_speed.linear.x = maxSpeed
                new_speed.angular.z = 0.0
            if averageLeft < desRangeLeftMin:
                # wall is too close: rotate away from wall
                rospy.loginfo("Wall is too close")
                new_speed.linear.x = maxSpeed
                new_speed.angular.z = maxRotateSpeed
            if averageLeft > desRangeLeftMax:
                # wall is too far: rotate towards wall
                rospy.loginfo("Wall is too far")
                new_speed.linear.x = maxSpeed
                new_speed.angular.z = -(maxRotateSpeed)

            if findGreater(laser_data.ranges, firstLower, firstHigher, secondLower, secondHigher) < -1:
                rospy.loginfo("Detected convex wall")
                new_speed.linear.x = maxSpeed
                new_speed.angular.z = -(maxRotateSpeed)
        else:
            # No wall to our left
            new_speed.linear.x = maxSpeed
            if detectedWall > 0:
                detectedWall -= 1
            if detectedWall > 25:
                new_speed.angular.z = -maxRotateSpeed * 1.5

    pub.publish(new_speed)


if __name__ == '__main__':
    detectedWall = 0
    rospy.init_node('obstacle_stopper')

    rospy.Subscriber('base_scan', LaserScan, laser_callback)
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)
    rospy.spin()
