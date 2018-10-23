#include "config.h"
#include "stitching.h"
#include "SLAM.h"
#include "ros/ros.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <cassert>
#include <stdio.h>
#include "pointmatcher/PointMatcher.h"
#include "exercise3/stitching_points.h"

const float PADDING = -1.0;
// minimum number of 'white' cells before performing stitching
// on the simulator: a normal local map will have from ~200 to ~400 white pixels
const unsigned min_white = 150;
//the threshold for which we consider a point 'white'
const float white_threshold = 0.7;

// maximum distance for an edge fitting before discarding the result and not stitching
const float edge_max_trans = square(2*grid_resolution); // squared metres
// maximum rotation for an edge fitting before discarding the result and not stitching
const float edge_max_theta = to_radians(20); // radians
// minimum absolute stitching angle before paying attention to the result of ICP
const float edge_ignore_threshold = to_radians(2);
// minimum number of points required before doing edge stitching
const unsigned min_points = 400;

// parameters
// alpha = weight for squared translational offset
// beta  = weight for squared rotational offset
// chi   = weight for sum of absolute difference in the map
#ifdef SIMULATOR
    #if OCCUPANCY == OCC_AABB
        const double alpha = 1.0, beta = 30.0, chi = 1.0;
    #elif OCCUPANCY == OCC_ANN
        const double alpha = 1.0, beta = 100.0, chi = 1.0;
    #elif OCCUPANCY == OCC_STD
        const double alpha = 1.0, beta = 100.0, chi = 1.0;
    #endif
#else // real robot
    #if OCCUPANCY == OCC_AABB
        const double alpha = 1.0, beta = 55.0, chi = 1.0;
    #elif OCCUPANCY == OCC_ANN
        const double alpha = 1.0, beta = 0.0, chi = 1.0;
    #elif OCCUPANCY == OCC_STD
        const double alpha = 0.0, beta = 200.0, chi = 1.0;
    #endif
#endif



/// draw an occupancy grid to stdout
void OccGrid::print() {
    for(unsigned y = 0; y < height; ++y) {
        for(unsigned x = 0; x < width; ++x) {
            float cell = getCell(x, y);
            if(x == robot_x && y == robot_y) {
                std::cout << "X ";
            } else if (cell == PADDING) {
                std::cout << ":)";
            } else {
                std::cout << cell << " ";
            }
        }
        std::cout << "\n";
    }
}

exercise3::local_map OccGrid::getMsg(unsigned counter) {
    exercise3::local_map msg;
    msg.dx = 0;
    msg.dy = 0;
    msg.dtheta = 0;
    msg.resolution = grid_resolution;
    msg.cols = width;
    msg.counter = counter;
    //msg.timestamp = 0;
    msg.local_map.resize(width*height);
    for(unsigned i = 0; i < width*height; ++i)
        msg.local_map[i] = cells[i];
    return msg;
}

bool OccGrid::enoughWhite(unsigned minimum) {
    unsigned white_count = 0; // total number of 'white' cells in local mp

    for(unsigned y = 0; y < height ; ++y) {
        for(unsigned x = 0; x < width; ++x) {
            float currentCell = getCell(x,y);

            if(currentCell != PADDING && currentCell >= white_threshold) {
                ++white_count;

                // no point continuing to count them
                if(white_count >= minimum) {
                    return true;
                }
            }
        }
    }

    //ROS_INFO("num white %d", white_count);

    return white_count >= minimum; // required for minimum=0 case
}


/** Returns a value indicating the cost from superimposing the local map
 occupancy grid onto the global map at the current estimated position of the
 robot, translated by some offset number of cells (xOffset, yOffset). Eg no
 offset (0, 0) would place the bottom-middle of the local map directly on top
 of the estimated robot position.
 */
double OccGrid::costFunction(OccGrid *globalMap, int xOffset, int yOffset, double angleOffset) {
    //ROS_INFO("stitching params: %f %f %f", alpha, beta, chi);

    // Sum of squared difference between x and y
    double firstCost = xOffset*xOffset + yOffset*yOffset;

    // Calculate angle difference
		//std::cout << "HEYYY" << robot_orientation << std::endl;
    //double secondCost = std::abs((globalMap->robot_orientation + robot_orientation) - globalMap->robot_orientation);
    double secondCost = fabs(angleOffset);

    // Calculate the overlay between actual position
    // (with x_offset and y_offset offset applied)
    // on global map and actual position on local map
    int fromX = (globalMap->robot_x + xOffset) - robot_x;
    int toX   = fromX + width;
    int fromY = (globalMap->robot_y + yOffset) - robot_y;
    int toY   = fromY + height;

    double thirdCost = 0.0;
    // indices into the original localMap
    unsigned counterX = 0;
    unsigned counterY = 0;

    for (int x = fromX; x < toX; ++x) {
        if (x < 0 || x >= int(globalMap->width)) {
            ++counterX;
            continue;
        }
        for (int y = fromY; y < toY; ++y) {
            if (y < 0 || y >= int(globalMap->height)) {
                ++counterY;
                continue;
            }
            float val = getCell(counterX, counterY);
            if(val != PADDING) {
                thirdCost += std::abs(globalMap->getCell(x, y) - val);
            }
            ++counterY;
        }
        counterY = 0;
        ++counterX;
    }
    //ROS_INFO("first cost = %f, second cost = %f, third cost = %f", firstCost, secondCost, thirdCost);
    return (alpha * firstCost) + (beta * secondCost) + (chi * thirdCost);
}

/** Rotates a _pre-padded_ occupancy grid by `rotationAmount` (radians) about the
 given origin cell on the local map: (xOrigin, yOrigin). Writes the result
 into OccGrid `out` padding cells around the edge are indicated with the
 value -1 (PADDING)
 */
void OccGrid::rotated(int xOrigin, int yOrigin, double rotationAmount, OccGrid *out) const {
    // could be outside range of original map, but not useful for us
    assert(0 <= xOrigin && xOrigin < int(width) &&
           0 <= yOrigin && yOrigin < int(height));
    assert(out != NULL);
    assert(getSize() == out->getSize()); // should both be padded

    out->setAllCells(PADDING);

    // loop over the cells of the newly rotated map (out)
    for (int y = 0; y < int(height); ++y) {
        for (int x = 0; x < int(width); ++x) {
            // vector from the cell to the centre of rotation
            float diffX = xOrigin - x;
            float diffY = yOrigin - y;

            // polar vector from the cell to the centre of rotation
            float angle = atan2(diffY, diffX);
            float hypot = sqrt(diffX*diffX + diffY*diffY);

            angle -= rotationAmount;

            // (approximate) x, y of the cell with rotation applied on the original map
            int newDiffX = int(xOrigin) - round((cos(angle) * hypot));
            int newDiffY = int(yOrigin) - round((sin(angle) * hypot));

            if(newDiffX >= 0 && newDiffX < int(width) &&
               newDiffY >= 0 && newDiffY < int(height)) {
                float cell = getCell(newDiffX, newDiffY);
                if(cell != PADDING) {
                    out->setCell(x, y, cell);
                }
            }
        }
    }

    out->robot_x = robot_x;
    out->robot_y = robot_y;
    // normalise angle (always positive)
    out->robot_orientation = mod(robot_orientation + rotationAmount, tau);
}

/** Finds the optimal rotation and x,y translation that minimises the cost
 function of overlaying the transformed local map onto the global map.
 The local map should be pre-padded.
 tmp should contain two pre-allocated padded OccGrid's. A pointer to the
 optimal one will be written to the return value.
 The global map should have its robot pose estimate updated by the latest
 available odometry
 */
FittedMap OccGrid::optimalFit(OccGrid *globalMap, DoubleBuffer<OccGrid> tmp) {
    // should all be allocated and pre-padded
    assert(tmp.a->getSize() == tmp.b->getSize() &&
           tmp.a->getSize() == getSize());

    double startingAngle = globalMap->robot_orientation; //The starting angle of the adjustments to be made

    if(!enoughWhite(min_white)) { // don't do stitching
        rotatedAboutRobot(startingAngle, tmp.cursor);
        std::cout << "Not enough information to stitch." << std::endl;
        return FittedMap(0, 0, 0, 1, globalMap, tmp.cursor);
    }

    double currentCost = 1000000;
    double a_best = 0;
    double x_best = 0;
    double y_best = 0;

    OccGrid *currentBest = this; // so as to never return NULL

    for(int degrees = -5; degrees <= 5; ++degrees) {
        double angle = to_radians(degrees);
        rotatedAboutRobot(startingAngle + angle, tmp.cursor);
        //ROS_INFO("angle = %f corresponds = %f", angle, startingAngle+angle);
        //tmp.cursor->print();

        bool this_angle_best = false;

        for (int x = 0; x <= 0; x += 1) {
            for (int y = 0; y <= 0; y += 1) {
                double value = tmp.cursor->costFunction(globalMap, x, y, angle);
                //std::cout
                    //<< "COST FOR " << x << "," << y << "," << angle << ": "
                    //<< value << "," << std::endl;
                if (value < currentCost) {
                    //ROS_INFO("best now %p", tmp.cursor);
                    currentCost = value;
                    currentBest = tmp.cursor;
                    a_best = angle;
                    x_best = x;
                    y_best = y;
                    this_angle_best = true;
                }
            }
        }

        if(this_angle_best)
            tmp.switch_buffers(); // start writing to the other buffer
    }

    //std::cout << "Found best angle to be " << a_best << std::endl;
    //std::cout << "Found translation to be " << x_best << ", " << y_best << std::endl;

    //ROS_INFO("DONE angle=%f, offset=(%f, %f)", a_best, x_best, y_best);

    return FittedMap(x_best, y_best, a_best, currentCost, globalMap, currentBest);
}


FittedMap OccGrid::gradientDescentFit(OccGrid *globalMap, DoubleBuffer<OccGrid> tmp) {
    // should all be allocated and pre-padded
    assert(tmp.a->getSize() == tmp.b->getSize() &&
           tmp.a->getSize() == getSize());

    double est_angle = globalMap->robot_orientation;
    double current_angle = est_angle;
    double delta; // gradient (pretend initialisation)
    double eta = 0.01; // descent steepness
    double cost;

    double last_angle = 0;
    double last_cost  = -1;

    // number of iterations
    for(unsigned i = 0; i < 20; ++i) {
        double current_norm_angle = mod(current_angle, tau);
        rotatedAboutRobot(current_norm_angle, tmp.cursor);
        cost = tmp.cursor->costFunction(globalMap, 0, 0, fabs(current_angle-est_angle)) / 100;
        std::cout<< "(" << current_angle << ", " << cost << "),\n";

        // this is the first test
        if(last_cost == -1) {
            last_angle = current_angle;
            last_cost = cost;
            current_angle += 0.1;
            continue;
        }

        if(fabs(current_angle-last_angle) < 0.005) // pretty much converged
            break;

        delta = (cost-last_cost) / (current_angle-last_angle);
        //std::cout<< delta << "\n";

        last_angle = current_angle;
        current_angle = current_angle - eta*delta;
        last_cost = cost;

        eta *= 0.85; // decay steepness,
    }

    return FittedMap(0, 0, current_angle-est_angle, cost, globalMap, tmp.cursor);
}




void get_points(OccGrid *globalMap, std::vector<Point> *out) {
    // cell location of the robot
    int rx = globalMap->robot_x;
    int ry = globalMap->robot_y;

    // either have to take orientation into account, or take a square
    int radius = l_grid_y + 4;

    int start_y = clamp(ry - radius, 0, globalMap->height-1);
    int stop_y  = clamp(ry + radius, 0, globalMap->height-1);
    int start_x = clamp(rx - radius, 0, globalMap->width-1);
    int stop_x  = clamp(rx + radius, 0, globalMap->width-1);

    //ROS_INFO("points from %d,%d to %d,%d", start_x, start_y, stop_x, stop_y);

    for(int y = start_y; y < stop_y; ++y) {
        for(int x = start_x; x < stop_x; ++x) {
            if(globalMap->getCell(x, y) >= white_threshold) {
                out->push_back(distance_from_cell_top_left(rx, ry, x, y));
            }
        }
    }

    //ROS_INFO("%ld points", out->size());
}


void custom_icp(PointMatcher<float>::ICP &icp) {
    typedef PointMatcher<float> Matcher;

    // setup parameters in memory without YAML
    // https://github.com/ethz-asl/libpointmatcher/blob/master/doc/icpWithoutYaml.md
    //
    // also: https://libpointmatcher.readthedocs.io/en/stable/ICPIntro/
    // values documentation: http://wiki.ros.org/ethzasl_icp_configuration#PointToPlaneErrorMinimizer
    Matcher::Parameters params;
    std::string name;

    // reference filter
    name = "SamplingSurfaceNormalDataPointsFilter";
    params["knn"] = "8";
    params["ratio"] = "0.85"; // ratio to keep
    Matcher::DataPointsFilter* ref_filter =
        Matcher::get().DataPointsFilterRegistrar.create(name, params);
    params.clear();

    // construct matcher (KDtree matcher)
    name = "KDTreeMatcher";
    params["knn"] = "1";
    params["epsilon"] = "0";
    //params["maxDist"] = "1";
    Matcher::Matcher *kdtree =
        Matcher::get().MatcherRegistrar.create(name, params);
    params.clear();


    // outlier filter
    name = "TrimmedDistOutlierFilter";
    // ratio = proportion to keep
    params["ratio"] = "0.90";
    Matcher::OutlierFilter *trim =
        Matcher::get().OutlierFilterRegistrar.create(name, params);
    params.clear();

    // construct error minimiser
    name = "PointToPlaneErrorMinimizer";
    params["force2D"] = "1";
    Matcher::ErrorMinimizer *minimiser =
        Matcher::get().ErrorMinimizerRegistrar.create(name, params);
    params.clear();

    // transformation checkers (stopping conditions)
    name = "CounterTransformationChecker";
    params["maxIterationCount"] = "40";
    Matcher::TransformationChecker *maxIter =
        Matcher::get().TransformationCheckerRegistrar.create(name, params);
    params.clear();

    // stop once over 'smoothLength' iterations the points do not change more
    // than the given amounts
    name = "DifferentialTransformationChecker";
    params["minDiffRotErr"] = "0.01";
    params["minDiffTransErr"] = "0.001";
    params["smoothLength"] = "4";
    Matcher::TransformationChecker *diff =
        Matcher::get().TransformationCheckerRegistrar.create(name, params);
    params.clear();

    // Prepare inspector
    name = "NullInspector";
    Matcher::Inspector *inspector =
        Matcher::get().InspectorRegistrar.create(name, params);
    params.clear();

    // specify transformation
    // rigid/Euclidean transformation is rotation and translation only
    Matcher::Transformation *rigidTrans =
        Matcher::get().TransformationRegistrar.create("RigidTransformation");

    icp.referenceDataPointsFilters.push_back(ref_filter);
    icp.matcher.reset(kdtree);
    icp.outlierFilters.push_back(trim);
    icp.errorMinimizer.reset(minimiser);
    icp.transformationCheckers.push_back(maxIter);
    icp.transformationCheckers.push_back(diff);
    icp.inspector.reset(inspector);
    icp.transformations.push_back(rigidTrans);
}


FittedMap OccGrid::edgeFit(OccGrid *globalMap, DoubleBuffer<OccGrid> tmp,
        const sensor_msgs::LaserScan::ConstPtr &last_laser_data,
        exercise3::stitching_points *msg) {

    // should all be allocated and pre-padded
    assert(tmp.a->getSize() == tmp.b->getSize() &&
           tmp.a->getSize() == getSize());

    const float *ranges = &last_laser_data->ranges[0];

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> VertexMat;

#ifdef ENABLE_EXPERIMENTS

    if(counter == 10) {
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        ROS_INFO("SPECIAL CASE");
        double angle = to_radians(0);
        rotatedAboutRobot(globalMap->robot_orientation + angle, tmp.cursor);
        counter++;
        return FittedMap(100, 100, angle, 1, globalMap, tmp.cursor);
    } else {
        // perform no stitching
        rotatedAboutRobot(globalMap->robot_orientation, tmp.cursor);
        counter++;
        return FittedMap(0, 0, 0, 1, globalMap, tmp.cursor);
    }

#endif



// Obtain global map

    // polar coordinates local to the robot
    std::vector<Point> edge_points;
    edge_points.reserve(1000); // preallocate about the maximum expected number
    get_points(globalMap, &edge_points);

    if(edge_points.size() < min_points) {
        rotatedAboutRobot(globalMap->robot_orientation, tmp.cursor);
        std::cout << "Not enough information to stitch." << std::endl;
        return FittedMap(0, 0, 0, 1, globalMap, tmp.cursor);
    }
    ROS_INFO("%ld POINTS", edge_points.size());

    // the points of the global map are the target (to be matched to)
    VertexMat reference(3, edge_points.size());

    msg->rx = 0;
    msg->ry = 0;
    msg->global_xs.resize(edge_points.size());
    msg->global_ys.resize(edge_points.size());
    for(unsigned i = 0; i < edge_points.size(); ++i) {
        Point p = edge_points[i];
        msg->global_xs[i] = p.x;
        msg->global_ys[i] = p.y;

        reference.col(i) << p.x, p.y, 1;
    }


// Obtain local map

    // not all the lasers will be stored (because they are out of range)
    msg->local_xs.reserve(num_lasers);
    msg->local_ys.reserve(num_lasers);
    float rot = globalMap->robot_orientation;
    float max_range = last_laser_data->range_max;
    for(unsigned i = 0; i < num_lasers; ++i) {
        if(ranges[i] >= max_range)
            continue;

        Pol p(ranges[i], laser_angle(i) - rot);

        Point pxy = p.to_point();
        msg->local_xs.push_back(pxy.x);
        msg->local_ys.push_back(pxy.y);
    }

    unsigned num_used_lasers = msg->local_xs.size();
    // the points of the local map are the source (to be rigidly transformed)
    VertexMat source(3, num_used_lasers);

    for(unsigned i = 0; i < num_used_lasers; ++i) {
        source.col(i) << msg->local_xs[i], msg->local_ys[i], 1;
    }




// perform ICP

    // a point cloud = 'feature matrix' is an Eigen matrix with M rows, one for
    // each 'feature' (x,y,z) and N columns (number of points)
    typedef PointMatcher<float> Matcher;
    typedef Matcher::DataPoints::Label Label;

    Matcher::DataPoints::Labels dimensions; // inherited from std::vector<Label>
    dimensions.push_back(Label("x", 1)); // 1 = span
    dimensions.push_back(Label("y", 1)); // 1 = span
    dimensions.push_back(Label("z", 1)); // 1 = span

    Matcher::DataPoints source_points(source, dimensions);
    Matcher::DataPoints reference_points(reference, dimensions);


    Matcher::ICP icp;
    //icp.setDefault(); // default implementation
    custom_icp(icp);


    // empty cloud => std::runtime_error

    Matcher::TransformationParameters T;
    try{
        T = icp(source_points, reference_points);
    } catch (Matcher::ConvergenceError error) {
		ROS_ERROR("ICP failed to converge: %s", error.what());

        // perform no stitching
        rotatedAboutRobot(globalMap->robot_orientation, tmp.cursor);
        return FittedMap(0, 0, 0, 1, globalMap, tmp.cursor);
	}


// extract transformation
    // http://math.stackexchange.com/a/417813
    float tx = T(0, 2);
    float ty = T(1, 2);
    float s_x_cos_theta = T(0, 0);
    float theta;

    // sometimes the rotation matrix is a bit wrong (T(0,0) is too large)
    // ignore small errors
    if(fabs(s_x_cos_theta) < edge_ignore_threshold)
        theta = 0;
    else {
        // might have some scaling so solve for s_x
        float s_x_sin_theta = T(0, 1);
        // tan(theta) = sin(theta)/cos(theta) = s_x_sin_theta/s_x_cos_theta
        theta = atan2(s_x_sin_theta, s_x_cos_theta);
    }

    // translation is given relative to the robot orientation, but the stitcher
    // should return translation relative to the global map (0 rotation)
    Pol trans = Point(tx, ty).to_polar();
    trans.theta += globalMap->robot_orientation;
    Point converted_trans = trans.to_point();
    tx = converted_trans.x;
    ty = converted_trans.y;


    std::cout<< "points used " << icp.getReadingFiltered().features.cols() << "\n";
    std::cout<< "used ratio: " << icp.errorMinimizer->getWeightedPointUsedRatio() << "\n";
    std::cout<< "rotation: " << theta << "\n";
    std::cout<< "translation: " << tx << ", " << ty << "\n";
    std::cout<< "transformation:\n" << T << "\n";


    // if transformation too crazy, ignore it
    if(square(tx) + square(ty) > edge_max_trans || fabs(theta) > edge_max_theta) {
        // perform no stitching
        ROS_INFO("transformation too crazy: not stitching");
        rotatedAboutRobot(globalMap->robot_orientation, tmp.cursor);
        return FittedMap(0, 0, 0, 1, globalMap, tmp.cursor);
    }


// Write transformed points to message

    source = T*source;
    msg->local_trans_xs.resize(num_used_lasers);
    msg->local_trans_ys.resize(num_used_lasers);
    for(unsigned i = 0; i < num_used_lasers; ++i) {
        msg->local_trans_xs[i] = source(0, i);
        msg->local_trans_ys[i] = source(1, i);
        //assert(fabs(source(2, i)) < 0.001); // shouldn't be any z component
    }
    Eigen::Vector3f robot_pos;
    robot_pos<< 0, 0, 1;
    robot_pos = T*robot_pos;
    msg->trans_rx = robot_pos(0, 0);
    msg->trans_ry = robot_pos(1, 0);


    int tx_cells = int(round(tx/grid_resolution));
    int ty_cells = int(round(ty/grid_resolution));
    rotatedAboutRobot(globalMap->robot_orientation + theta, tmp.cursor);
    return FittedMap(tx_cells, ty_cells, theta, 0, globalMap, tmp.cursor);
}



/** writes a map padded with enough space to allow for any rotation about the
    robot to `out` with -1's (=PADDING) used for padding
 */
void OccGrid::padded(OccGrid *out) const {
    assert(out != NULL);
    assert(getPaddedSize() == out->getSize());

    out->setAllCells(PADDING);

    // starting values for the write cursor into `out`
    unsigned xOffset = (out->width / 2) - (width / 2);
    unsigned yOffset = (out->height / 2) - height;

    for(unsigned y = 0; y < height; ++y) {
        for(unsigned x = 0; x < width; ++x) {
            out->setCell(xOffset + x, yOffset + y, getCell(x, y));
        }
    }

    out->robot_x = robot_x + xOffset;
    out->robot_y = robot_y + yOffset;
    out->robot_orientation = robot_orientation;
}

/*
 int main(int argc, const char * argv[]) {
 OccGrid *local = new OccGrid();
 local->height = 23;
 local->width = 46;
 local->cells = new float[local->height * local->width];;
 local->setAllCells(-1);
 local->robot_x = local->width / 2;
 local->robot_y = local->height - 1;
 for(unsigned y = 0; y < local->height; ++y) {
 for(unsigned x = 0; x < local->width; ++x) {
 if (x < local->width / 2) {
 local->setCell(x, y, 1);
 } else {
 local->setCell(x, y, 9);
 }
 };
 }
 local->print();

 OccGrid *newLocal = new OccGrid();
 newLocal->padded(local);

 newLocal->print();

 OccGrid *newOne = new OccGrid();
 newOne->height = 23;
 newOne->width = 46;
 newOne->cells = new float[newOne->height * newOne->width];
 newOne->setAllCells(-1);


 std::cout << "\n";

 float rad = 2 * (M_PI / 180);
 local->rotatedAboutRobot(rad, newOne);
 newOne->print();

 return 0;
 } */
