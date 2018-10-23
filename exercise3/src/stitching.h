#ifndef STITCHING_H
#define STITCHING_H

#include <cstdlib>
#include "SLAM.h"
#include "exercise3/local_map.h"
#include "exercise3/stitching_points.h"


// forward declarations
struct Size;
struct OccGrid;
struct FittedMap;

// defined so sizes of different OccGrid's can be compared
struct Size {
    unsigned width, height;
    Size(unsigned width, unsigned height) : width(width), height(height) {}
    bool operator==(const Size &other)
        { return width == other.width  &&  height == other.height; }
};

template<typename T>
struct DoubleBuffer {
    T *a, *b;
    T *cursor;
    DoubleBuffer(T *a, T *b) : a(a), b(b), cursor(a) {}
    inline void switch_buffers() { if(cursor == a) cursor = b; else cursor = a; }
};


// local or global occupancy grid
struct OccGrid {
    unsigned robot_x, robot_y; // in cells
    double robot_orientation; // in radians, normalised but may be negative
    unsigned width, height; // number of cells
    float *cells; // does not own the cells so does not have to free them

    OccGrid() : width(0), height(0), cells(NULL) {}

    inline void setAllCells(float val) {
        for(unsigned i = 0; i < width*height; ++i)
            cells[i] = val;
    }
    inline float &operator()(unsigned x, unsigned y) { return cells[y*width+x]; }
    inline float getCell(unsigned x, unsigned y) const { return cells[y*width+x]; }
    inline void setCell(unsigned x, unsigned y, float val) { cells[y*width+x] = val; }
    inline Size getSize() const { return Size(width, height); }

    void print();
    exercise3::local_map getMsg(unsigned counter=0);

    bool enoughWhite(unsigned minimum);

    double costFunction(OccGrid *globalMap, int xOffset, int yOffset, double angleOffset);

    inline Size getPaddedSize() const {
        int hw = width/2;
        double diagonal = sqrt(height*height + hw*hw);
        unsigned dim = 2 * diagonal + 1;
        return Size(dim, dim);
    }
    void padded(OccGrid *out) const; // write a padded version out to the given OccGrid

    inline void rotatedAboutRobot(double rotationAmount, OccGrid *out) const {
        rotated(robot_x, robot_y, rotationAmount, out);
    }
    void rotated(int xOrigin, int yOrigin, double rotationAmount, OccGrid *out) const;

    FittedMap optimalFit(OccGrid *globalMap, DoubleBuffer<OccGrid> tmp);
    FittedMap gradientDescentFit(OccGrid *globalMap, DoubleBuffer<OccGrid> tmp);
    FittedMap edgeFit(OccGrid *globalMap, DoubleBuffer<OccGrid> tmp,
            const sensor_msgs::LaserScan::ConstPtr &ranges, exercise3::stitching_points *msg);
};

/** returned from findOptimalTranslation to contain the optimal parameters for
    transforming a local map to fit onto the global map, and gives the data for
    the local map when rotated by the optimal amount
*/
struct FittedMap {
    // the offset (in cells) from the odometry reported top left corner of the
    // local map
    int dx, dy;
    // the offset (in radians) from the odometry reported robot orientation.
    // the local map `map` is pre-rotated by
    // globaMap->robot_orientation (before stitching) + drot
    // aka
    // globalMap->robot_orientation (after stitching)
    double drot;
    double cost; // value of the local minimum of the cost function
    OccGrid *map; // rotation already applied

    FittedMap() {}

    FittedMap(unsigned dx, unsigned dy, double drot, double cost,
            OccGrid *globalMap, OccGrid *map)
        : dx(dx), dy(dy), drot(mod(drot, tau)), cost(cost), map(map) { }
};

#endif
