
#ifndef INV_SENSOR_H
#define INV_SENSOR_H

#include "config.h"
#include "exercise3/local_map.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Quaternion.h"
#include "nav_msgs/Odometry.h"
#include <ctime>
#include <algorithm>
#include <boost/thread.hpp>

// typedefs
typedef float  float32;
typedef double float64;
// boo, can't use c++11 due to incompatible ABI with ROS :(
#define nullptr NULL

// forward declarations
struct Pol;
struct Point;
void ANN_occupancy(const sensor_msgs::LaserScan::ConstPtr &last_laser_data);
void AABB_occupancy(const sensor_msgs::LaserScan::ConstPtr &last_laser_data);
void std_occupancy(const sensor_msgs::LaserScan::ConstPtr &last_laser_data);
void init_ANN();
float square(float);
double square(double);
double to_degrees(double);
double to_radians(double);

// parameters

const unsigned g_grid_x = 1000; // global grid x (width) in cells, should be even so the robot sits between two cells
const unsigned g_grid_y = 1000; // global grid y (height) in cells
#if OCCUPANCY == OCC_AABB || OCCUPANCY == OCC_STD
const float32  grid_resolution = 0.1; // metres/cell (cell size in metres) for both the local and global maps
#elif OCCUPANCY == OCC_ANN
const float32  grid_resolution = 0.25; // metres/cell (cell size in metres) for both the local and global maps
#endif

const double laser_range = 5.75; // metres
const unsigned l_grid_x = laser_range*2 / grid_resolution; // local grid x (width) in cells, should be even so the robot sits between two cells
const unsigned l_grid_y = laser_range / grid_resolution; // local grid y (height) in cells

// thresholds to determine whether the robot has moved sufficiently since the
// last local map
const float32 sufficient_translate = square(0.25); // squared metres
const float32 sufficient_rotate    = to_radians(20);

// how often to run stitching
const unsigned stitching_N = 4;


// globals
extern exercise3::local_map local_map;
extern Pol cell_locations[l_grid_y*l_grid_x]; // row major

const unsigned num_lasers = 512;
const double pi = 3.14159265358979323;
const double tau = 2*pi;



// helper functions
inline float32 square(float32 x) { return x*x; } // faster than pow(x, 2)
inline float64 square(float64 x) { return x*x; }
inline int clamp(int val, int min_val, int max_val) {
    return std::min(std::max(val, min_val), max_val);
}
inline double to_degrees(double radians) { return radians * 180.0 / pi; }
inline double to_radians(double degrees) { return degrees * pi / 180.0; }
inline double duration_since(clock_t start) {
    return double(clock() - start) / CLOCKS_PER_SEC * 1000;
}
inline double mod(double a, double b) {
    double res = fmod(a, b);
    if(res < 0)
        res += b;
    return res;
}


inline double laser_angle(unsigned number) {
    // exercise1 movement.py get_laser_xy does something different
    // reading 0 is the far right, reading 511 is the far left
    // the angle goes from 0 to pi
    return (double(number)/double(num_lasers)) * pi;
}

// types
struct Point {
    float32 x, y;

    Point() : x(0), y(0) {}
    Point(float32 set_x, float32 set_y) : x(set_x), y(set_y) {}
    inline Pol to_polar();
};
struct Pol {
    float32 r, theta;

    Pol() : r(0), theta(0) {}
    Pol(float32 set_r, float32 set_theta) : r(set_r), theta(set_theta) {}
    inline Point to_point() {
        return Point(r*cos(theta), r*sin(theta));
    }
};
// needs to be placed below the definition of Pol
inline Pol Point::to_polar() {
    return Pol(sqrt(x*x + y*y), atan2(y, x));
}


// from exercise2 pf_localisation.util.py
inline double quaternion_yaw(const geometry_msgs::Quaternion q) {
    return atan2(2 * (q.x * q.y + q.w * q.z),
                     q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);
}
struct Pose {
    float32 x, y, theta;

    Pose() : x(0), y(0), theta(0) {}
    Pose(float32 set_x, float32 set_y, float32 set_theta)
        : x(set_x), y(set_y), theta(set_theta) {}

    // extract information from an Odometry message
    // caller should hold odom_mutex
    Pose(const nav_msgs::Odometry* o) {
        x     = o->pose.pose.position.x;
        y     = o->pose.pose.position.y;
        theta = quaternion_yaw(o->pose.pose.orientation);
    }
    inline Point to_point() {
        return Point(x, y);
    }
    inline Pose &operator-=(const Pose &other) {
        x -= other.x; y -= other.y; theta -= other.theta;
        return *this;
    }
};
inline Pose operator-(const Pose& a, const Pose& b) {
    return Pose(a.x-b.x, a.y-b.y, a.theta-b.theta);
}


template<typename T>
struct RosMessageQueue {
    typedef typename T::ConstPtr ConstPtr;

    boost::mutex mutex;
    ConstPtr last_data; // initialised to nullptr

    // crucially: makes a _copy_ of the shared pointer so the global one can
    // change freely
    ConstPtr get_latest() {
        boost::unique_lock<boost::mutex> lock(mutex);
        return last_data; // increases shared counter
    }
    void set_latest(const ConstPtr &ptr) {
        boost::unique_lock<boost::mutex> lock(mutex);
        last_data = ptr; // increases shared counter
    }
};



// get the polar vector from the top left of the 'ref' cell to the center of the other cell in metres
inline Point distance_from_cell_top_left(int ref_x, int ref_y, int x, int y) {
    // a cell 1 cell to the right of the ref cell would have a distance of 1.5
    // cells in the x axis to reach the center of the other cell, hence
    // +0.5*resolution in the x axis
    //
    // a cell 1 cell above the ref cell would have a distance of 0.5 cells in
    // the y to reach the center of the cell, hence -0.5*resolution in the y
    // axis
    float32 cell_x = (x - ref_x)  * grid_resolution + 0.5*grid_resolution;
    float32 cell_y = (ref_y  - y) * grid_resolution - 0.5*grid_resolution;
    return Point(cell_x, cell_y);
}



#endif
