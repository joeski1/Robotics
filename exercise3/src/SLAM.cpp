/*
    Inverse sensor model

    @author Matt
*/

#include "config.h"
#include <cstddef> // for standard sized types (eg uint8_t)
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm> // for std::max
#include <math.h>

// ROS
#include "ros/ros.h"

// messages
#include "geometry_msgs/Quaternion.h"
#include "nav_msgs/Odometry.h"
#include "exercise3/local_map.h"
#include "exercise3/global_map.h"
#include "exercise3/global_map_request.h"
#include "exercise3/pose_update.h"
#include "exercise3/stitching_points.h"

// local includes
#include "SLAM.h"
#include "stitching.h"
#include "update.h"




// typedefs
typedef boost::lock_guard<boost::mutex> guard;

// forward declarations
bool sufficiently_moved(const nav_msgs::Odometry*);
inline float64 quaternion_yaw(const geometry_msgs::Quaternion);


// types

// used for not freeing memory from a shared_ptr
struct null_deleter { void operator()(void const *) const { } };


// helper functions
// a short sleep :)
inline void nap() {
    boost::this_thread::sleep(boost::posix_time::milliseconds(1));
}




///////////////////////////////
// Globals
///////////////////////////////
// local map
ros::Publisher local_map_pub;
exercise3::local_map local_map;
// used to track ownership. Don't want to ever free
boost::shared_ptr<exercise3::local_map>
    local_map_ownership(&local_map, null_deleter());
// for stitching
OccGrid local_map_occ;
#if STITCHING == STITCH_EDGE
ros::Publisher points_pub;
#endif
// shared between the 3 maps below
boost::shared_array<float> stitching_memory;
OccGrid local_map_padded_occ;
OccGrid tmp_padded_occs[2];
DoubleBuffer<OccGrid> stitching_tmp(
        &tmp_padded_occs[0], &tmp_padded_occs[1]);


// global map
ros::Publisher global_map_pub;
exercise3::global_map global_map;
boost::shared_ptr<exercise3::global_map>
    global_map_ownership(&global_map, null_deleter());
OccGrid global_map_occ;
#if UPDATE == UPDATE_LOG_ODDS
OccGrid global_map_log_occ;
#endif
bool should_publish_global_map = false; // only when requested

// pose estimation
Pose last_pose; // pose when local_map was last published (used to determine sufficient movement)
exercise3::pose_update pose_update_msg;
ros::Publisher pose_update_pub;
Pose first_pose; // odometry pose as-of startup, subtract from every subsequent pose

// sensor messages
RosMessageQueue<sensor_msgs::LaserScan> laser_data;
RosMessageQueue<nav_msgs::Odometry> odom_data;
boost::condition_variable sufficient_movement_condition;

// hold the sequence numbers of the last_laser_data and last_odometry_data that
// were used last time a local map was generated
unsigned last_used_laser = 1;
unsigned last_used_odometry = 1;

// last odometry readings
const int initial_rx = g_grid_x/2; // robot starting location
const int initial_ry = g_grid_y/2; // robot starting location
double abs_last_x = 0;
double abs_last_y = 0;
double odom_last_theta = 0;


template<typename T>
void blocking_publish(const ros::Publisher &p, const boost::shared_ptr<T> &ownership) {
    // in-case the last message isn't finished for some reason
    while(!ownership.unique())
        nap();

    // increments shared pointer ownership count, so no longer unique
    p.publish(ownership); // non-blocking ?

    // wait for the message to be serialised, at which point it can be modified again
    // note: this is a complete hack
    while(!ownership.unique())
        nap();
}



//Calculates the difference between two angles
double angleDiff(double theta1, double theta2) {
  if(theta1 < 0)
    theta1 += tau;
  if(theta2 < 0)
    theta2 += tau;
  double a = mod(theta1-theta2, tau);
  double b = mod(theta2-theta1, tau);
  if(a < b)
    return a;
  return -b;
}


// Takes the latest odometry pose estimate, which is: x, y (meters) and angle (radians)
// (0, 0, 0) corresponds to the centre of the global map facing upwards
void update_pose_estimate(OccGrid *globalMap, Pose odometry_pose) {
    // Work out difference from the last odometry reading
    double odom_diff_x = abs_last_x - odometry_pose.x;
    double odom_diff_y = abs_last_y - odometry_pose.y;
    double odom_diff_theta = angleDiff(odom_last_theta, odometry_pose.theta);

    // Calculate the translation polar vector from last_odom -> current_odom
    double absolute_diff_r = sqrt(square(odom_diff_x) + square(odom_diff_y)); // Magnitude
    double absolute_diff_theta = mod(atan2(odom_diff_y, odom_diff_x), tau);   // Angle

    // Find the difference between the last_odom orientation and absolute_diff_theta.
    // Add this on to the robot's orientation before the update.
    // I.e. if the robot orientation was off by 90 degrees, rotate the absolute_diff vector by 90 degrees
    double absolute_diff_effective_angle = mod(globalMap->robot_orientation + angleDiff(odom_last_theta, absolute_diff_theta), tau);

    // Convert the absolute_diff vector transformed by the effective angle to a
    // Cartesian vector
    double poseDiffx = -sin(absolute_diff_effective_angle) * absolute_diff_r;
    double poseDiffy = cos(absolute_diff_effective_angle) * absolute_diff_r;

    // Apply calculated movement to global map's robot coords
    globalMap->robot_x += int(round(poseDiffx/grid_resolution));
    globalMap->robot_y += int(round(poseDiffy/grid_resolution));
    globalMap->robot_orientation = mod(globalMap->robot_orientation + odom_diff_theta, tau);

    // Update the 'last' odom values
    abs_last_x = odometry_pose.x;
    abs_last_y = odometry_pose.y;
    odom_last_theta = odometry_pose.theta;

}


// caller should hold odom_mutex and laser_mutex
void populate_and_publish_local_map() {
    nav_msgs::Odometry::ConstPtr latest_odometry = odom_data.get_latest();
    sensor_msgs::LaserScan::ConstPtr latest_laser = laser_data.get_latest();

    // cannot process until some messages have come through
    if(latest_laser == nullptr || latest_odometry == nullptr)
        return;

    // should not process until ros has had a chance to provide new data.
    // This may be needed because this function maxes out the cpu
    const unsigned this_laser = latest_laser->header.seq;
    const unsigned this_odometry = latest_odometry->header.seq;
    if(last_used_laser == this_laser || last_used_odometry == this_odometry)
        return;
    last_used_laser = this_laser;
    last_used_odometry = this_odometry;
    //ROS_INFO("this laser = %i this odom = %i", this_laser, this_odometry);

    ROS_INFO("\n%s", config_string);



    // calculate change in pose since last local map
    Pose current_pose(latest_odometry.get()); // convert

#ifdef START_UPWARDS
    current_pose -= first_pose; // make relative to (0, 0, 0)
#endif

    const Pose d_pose = current_pose - last_pose;
    local_map.dx      = d_pose.x;
    local_map.dy      = d_pose.y;
    local_map.dtheta  = d_pose.theta;

    last_pose = current_pose;


    local_map.timestamp = ros::Time::now();

    // calculate local map occupancy using an inverse sensor model
    clock_t overallStart = clock();
    clock_t start = overallStart;

#if OCCUPANCY == OCC_AABB
    AABB_occupancy(latest_laser);
#elif OCCUPANCY == OCC_ANN
    ANN_occupancy(latest_laser);
#elif OCCUPANCY == OCC_STD
    std_occupancy(latest_laser);
#else
#error
#endif

    ROS_INFO("took %f ms to calculate local map %i ", duration_since(start),
            local_map.counter);
    start = clock();


#ifdef PUBLISH_LOCAL_MAP
    //blocking_publish(local_map_pub, local_map_ownership);
#endif


    start = clock();
    ///////////////////////////
    // handle stitching
    ///////////////////////////

    // take the tight fitting local map, and copy it into a larger map with
    // padding around the outside, ready for rotating
    update_pose_estimate(&global_map_occ, current_pose);
    local_map_occ.padded(&local_map_padded_occ); // preallocated, just fill out
    //local_map_pub.publish(local_map_padded_occ.getMsg(local_map.counter)); // send to viewer
    FittedMap fit;

#if STITCHING == STITCH_OFF
    local_map_padded_occ.rotatedAboutRobot(
            global_map_occ.robot_orientation, &tmp_padded_occs[0]);
    fit = FittedMap(0, 0, 0, 0, &global_map_occ, &tmp_padded_occs[0]);
#else


    if(local_map.counter > 3 && local_map.counter % stitching_N == 0) {
#if STITCHING == STITCH_BRUTE
        fit = local_map_padded_occ.optimalFit(&global_map_occ, stitching_tmp);
#elif STITCHING == STITCH_GRAD
        fit = local_map_padded_occ.gradientDescentFit(&global_map_occ, stitching_tmp);
#elif STITCHING == STITCH_EDGE
        exercise3::stitching_points stitching_points;
        stitching_points.counter = local_map.counter;
        stitching_points.timestamp = ros::Time::now();

        fit = local_map_padded_occ.edgeFit(&global_map_occ, stitching_tmp,
                latest_laser, &stitching_points);

        points_pub.publish(stitching_points);
#endif
    } else {
        // do no stitching
        ROS_INFO("not stitching this time");
        local_map_padded_occ.rotatedAboutRobot(
                global_map_occ.robot_orientation, &tmp_padded_occs[0]);
        fit = FittedMap(0, 0, 0, 0, &global_map_occ, &tmp_padded_occs[0]);
    }
#endif

    ROS_INFO("fit cost = %f at %p", fit.cost, fit.map);

#ifdef PUBLISH_LOCAL_MAP
    local_map_pub.publish(fit.map->getMsg(local_map.counter)); // send to viewer
#endif

    global_map_occ.robot_x += fit.dx;
    global_map_occ.robot_y += fit.dy;
    global_map_occ.robot_orientation += fit.drot;

#if UPDATE == UPDATE_LOG_ODDS
    global_map_log_occ.robot_x = global_map_occ.robot_x;
    global_map_log_occ.robot_y = global_map_occ.robot_y;
    global_map_log_occ.robot_orientation = global_map_occ.robot_orientation;
#endif


    // robot is positioned at the top left of the cell robot_x and robot_y
    Pose estimated_pose(
            global_map_occ.robot_x*grid_resolution,
            global_map_occ.robot_y*grid_resolution,
            mod(global_map_occ.robot_orientation + pi/2, tau));
            //Adding 90 degrees here because map viewer seems to think 0 degrees is horizontally left

    ROS_INFO("estimated pose: %f, %f, %f", estimated_pose.x, estimated_pose.y,
            to_degrees(estimated_pose.theta));

    ROS_INFO("took %f ms to find optimal fit", duration_since(start));
    start = clock();

    ///////////////////////////
    // handle updating
    ///////////////////////////
#if UPDATE == UPDATE_FAKE
    fake_update_global_map(fit, &global_map_occ);
#elif UPDATE == UPDATE_PROB
    update_global_map(fit, &global_map_occ);
#elif UPDATE == UPDATE_LOG_ODDS
    log_odds_update_global_map(fit, &global_map_log_occ, &global_map_occ);
#endif

    ROS_INFO("took %f ms to update global map", duration_since(start));
    start = clock();

    // publish pose update
    pose_update_msg.x = estimated_pose.x;
    pose_update_msg.y = estimated_pose.y;
    pose_update_msg.rotation = estimated_pose.theta;
    pose_update_msg.timestamp = ros::Time::now();
    pose_update_msg.counter++;

    pose_update_pub.publish(pose_update_msg);

    // publish global map and pose update
    global_map.timestamp = ros::Time::now();
    global_map.pose = pose_update_msg;

    if(should_publish_global_map) {
        blocking_publish(global_map_pub, global_map_ownership);
    }
    global_map.counter++;


    local_map.counter++;

    ROS_INFO("took %f ms in total", duration_since(overallStart));
    ROS_INFO(" ");
}

// indefinitely wait for sufficient movement: then publish a new local map
void generate_local_maps() {
    while(ros::ok()) {
        {
            boost::unique_lock<boost::mutex> odom_lock(odom_data.mutex);
            // only use .last_data while holding the mutex
            while(odom_data.last_data == nullptr ||
                  !sufficiently_moved(odom_data.last_data.get())) {

                sufficient_movement_condition.wait(odom_lock);

                // the condition is also notified when shutting down
                if(!ros::ok())
                    return;
            }
        }
        // release the odometry lock so more messages can come in

        populate_and_publish_local_map();
    }
}


// caller should hold odom_mutex
// o should != nullptr
bool sufficiently_moved(const nav_msgs::Odometry* o) {
#ifdef START_UPWARDS
    const float64 current_x     = o->pose.pose.position.x - first_pose.x;
    const float64 current_y     = o->pose.pose.position.y - first_pose.y;
    const float64 current_theta = quaternion_yaw(o->pose.pose.orientation)
                                    - first_pose.theta;
#else
    const float64 current_x     = o->pose.pose.position.x;
    const float64 current_y     = o->pose.pose.position.y;
    const float64 current_theta = quaternion_yaw(o->pose.pose.orientation);
#endif

    const float64 rotation  = fabs(current_theta - last_pose.theta);
    const float32 sq_dist   = square(current_x - last_pose.x)
                            + square(current_y - last_pose.y);

    return sq_dist > sufficient_translate
       || rotation > sufficient_rotate;
}



void laser_callback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    // msg documentation: http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
    laser_data.set_latest(msg);
}
void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    // msg documentation: http://docs.ros.org/api/nav_msgs/html/msg/Odometry.html
    bool was_null;
    {
        boost::unique_lock<boost::mutex> odom_lock(odom_data.mutex);
        was_null = odom_data.last_data == nullptr;
        odom_data.last_data = msg; // increases shared counter. Data is not freed
    }

    // first odometry message
    if(was_null)
        first_pose = Pose(msg.get());

    if(was_null || sufficiently_moved(msg.get()))
        sufficient_movement_condition.notify_one();
}
void global_map_request_callback(const exercise3::global_map_request::ConstPtr& msg) {
    should_publish_global_map = true;
}


// do some calculations that only have to be done once
void precompute() {
    srand(time(NULL)); // random seed

    // pre-allocate and set constants
    local_map.local_map.resize(l_grid_x*l_grid_y);
    local_map.counter = 0;
    local_map.resolution = grid_resolution;
    local_map.cols = l_grid_x;

    // for stitching
    local_map_occ.width  = l_grid_x;
    local_map_occ.height = l_grid_y;
    local_map_occ.cells  = &local_map.local_map[0];
    local_map_occ.robot_x = l_grid_x/2;
    local_map_occ.robot_y = l_grid_y;
    local_map_occ.robot_orientation = 0;


    // allocate global map
    global_map.global_map.resize(g_grid_x*g_grid_y);
    for(unsigned i = 0; i < g_grid_x*g_grid_y; ++i)
        global_map.global_map[i] = 0.5; // initialise global map cells

    global_map.counter = 0;
    global_map.resolution = grid_resolution;
    global_map.cols = g_grid_x;

    global_map_occ.robot_x = initial_rx;
    global_map_occ.robot_y = initial_ry;

    global_map_occ.width  = g_grid_x;
    global_map_occ.height = g_grid_y;
    global_map_occ.cells  = &global_map.global_map[0];

#if UPDATE == UPDATE_LOG_ODDS
    global_map_log_occ.robot_x = initial_rx;
    global_map_log_occ.robot_y = initial_ry;

    global_map_log_occ.width  = g_grid_x;
    global_map_log_occ.height = g_grid_y;
    global_map_log_occ.cells  = new float[g_grid_x*g_grid_y]; // TODO: memory leak
    // prior = log(0.5/(1-0.5)) = log(1) = 0
    memset(global_map_log_occ.cells, 0, sizeof(float)*g_grid_x*g_grid_y);
#endif

    // setup pose update
    pose_update_msg.counter = 0;
    pose_update_msg.x = 0;
    pose_update_msg.y = 0;
    pose_update_msg.rotation = 0;

    // allocate data for stitching
    Size tmps = local_map_occ.getPaddedSize(); // size of the temporary local maps
    printf("Padded size: %i, %i\n", tmps.width, tmps.height);
    size_t tmp_cells = tmps.width * tmps.height;
    // require 3 temporary maps
    stitching_memory = boost::shared_array<float>(new float[3*tmp_cells]);

    // distribute the flat array between the 3 temporary grids
    local_map_padded_occ.width  = tmps.width;
    local_map_padded_occ.height = tmps.height;
    local_map_padded_occ.cells  = stitching_memory.get() + 0;

    tmp_padded_occs[0].width  = tmps.width;
    tmp_padded_occs[0].height = tmps.height;
    tmp_padded_occs[0].cells  = stitching_memory.get() + tmp_cells;

    tmp_padded_occs[1].width  = tmps.width;
    tmp_padded_occs[1].height = tmps.height;
    tmp_padded_occs[1].cells  = stitching_memory.get() + 2*tmp_cells;


    // initialise the neural network
    init_ANN();
}


int main(int argc, char **argv) {
    assert(sizeof(float32)*8 == 32);
    assert(sizeof(float64)*8 == 64);

    ros::init(argc, argv, "inv_sensor");
    ros::NodeHandle n;

    // cannot make publisher blocking, so instead just buffer several local
    // maps. They should all have consistent data (all from the same point in
    // time) so the processing of the local map can be done at a later date just
    // fine so long as none of the local maps go missing.
    local_map_pub = n.advertise<exercise3::local_map>("local_map", 100);

    global_map_pub = n.advertise<exercise3::global_map>("global_map", 100);

    pose_update_pub = n.advertise<exercise3::pose_update>("pose_update", 100);

#if STITCHING == STITCH_EDGE
    points_pub = n.advertise<exercise3::stitching_points>("local_map_points", 100);
#endif

    // queue size of 1 => always process the latest information available and
    // don't buffer any. It is important for odometry and laser data to be up to
    // date after waiting while the local map gets generated.
    ros::Subscriber odometry_sub =
        n.subscribe(ODOMETRY_CHANNEL, 1, odometry_callback);

    ros::Subscriber laser_sub =
        n.subscribe("base_scan", 1, laser_callback);

    ros::Subscriber global_map_request_sub =
        n.subscribe("global_map_request", 1, global_map_request_callback);

    precompute();

    boost::thread t(generate_local_maps);
    t.detach(); // don't intend to join, closes by itself

    ros::spin(); // process callbacks indefinitely (on this thread)

    // release the condition variable to make sure that the generate_local_maps
    // gets to shutdown gracefully
    {
        boost::unique_lock<boost::mutex> odom_lock(odom_data.mutex);
        sufficient_movement_condition.notify_all();
    }
    printf("shutting down inv_sensor\n");

    return 0;
}
