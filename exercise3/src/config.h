
// wrap questionable experiments in #ifdef this
//#define ENABLE_EXPERIMENTS


// whether to compile for running in the simulator
//#define SIMULATOR

// whether to send local map messages
#define PUBLISH_LOCAL_MAP


// local map occupancy by AABB method
#define OCC_AABB 1
// local map occupancy by neural network
#define OCC_ANN  2
// local map occupancy by the 'standard method' as described in probabilistic
// robotics chapter 9
#define OCC_STD  3

#define OCCUPANCY OCC_STD


// store the global map as log-odds to reduce the accumulated floating point error
#define UPDATE_FAKE     1
#define UPDATE_PROB     2
#define UPDATE_LOG_ODDS 3

#define UPDATE UPDATE_LOG_ODDS


// no stitching
#define STITCH_OFF   0
// stitching by brute force optimisation of sum-absolute-error
#define STITCH_BRUTE 1
// stitching by gradient decent optimisation of sum-absolute-error
#define STITCH_GRAD  2
// stitching by feature matching between the local and global maps
#define STITCH_EDGE  3


// simulator only options
#ifdef SIMULATOR

    // whether to bypass stitching and trust the odometry
    #define STITCHING STITCH_OFF

    // whether to solve the closed loop problem by introducing noise into the
    // perfect simulated odometry
    #define ODOMETRY_CHANNEL "odom"
    //#define ODOMETRY_CHANNEL "odom_noisy"

#else
    #define STITCHING STITCH_EDGE
    #define ODOMETRY_CHANNEL "odom"
#endif



// broken

// whether to subtract the very first odometry reading from every subsequent reading
//#define START_UPWARDS



#ifndef CONFIG_H
#define CONFIG_H

const char config_string[] =

#ifdef SIMULATOR
    "SIMULATOR"
#else
    "REAL"
#endif

    " - "

    "Odometry: '" ODOMETRY_CHANNEL "'"

    " - "

    "Occupancy: "
#if OCCUPANCY == OCC_AABB
    "AABB"
#elif OCCUPANCY == OCC_ANN
    "Neural Network"
#elif OCCUPANCY == OCC_STD
    "Standard"
#else
#   error
#endif

    " - "

    "Update: "
#if UPDATE == UPDATE_FAKE
    "Fake"
#elif UPDATE == UPDATE_PROB
    "Prob"
#elif UPDATE == UPDATE_LOG_ODDS
    "Log Odds"
#else
#   error
#endif

    " - "

    "Stitching: "
#if STITCHING == STITCH_OFF
    "OFF"
#elif STITCHING == STITCH_BRUTE
    "Brute force"
#elif STITCHING == STITCH_GRAD
    "Gradient Descent"
#elif STITCHING == STITCH_EDGE
    "Edges"
#else
#   error
#endif
    ;


#endif
