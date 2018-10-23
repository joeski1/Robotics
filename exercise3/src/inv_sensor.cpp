/*
    Inverse sensor models

    Several different approaches for forming local occupancy maps.
*/

#include "config.h"
#include "SLAM.h"
#include "ros/ros.h"
#include <cassert>
#include "Eigen/Dense"
#include <iostream>
// for parsing csv
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>

// scalar type
typedef double elem;

template<unsigned num_rows, unsigned num_cols>
struct Mat {
    typedef Eigen::Matrix<elem, num_rows, num_cols, Eigen::RowMajor> mat_type;
    typedef Eigen::Matrix<elem, num_rows, num_cols> single_row_mat_type;
    const static unsigned rows = num_rows;
    const static unsigned cols = num_cols;
    const static unsigned num_cells = num_rows * num_cols;
};



///////////////////////////////
// Neural Network Globals
///////////////////////////////

// network parameters
const unsigned readings_per_cell = 4;
const unsigned input_dimensions = readings_per_cell + 2;

//const unsigned hidden_neurons = 15;
//const unsigned hidden_neurons = 50;
const unsigned hidden_neurons = 100;

typedef Mat<hidden_neurons, input_dimensions> hidden_weights_type;
typedef Mat<1, hidden_neurons>                output_weights_type;

hidden_weights_type::mat_type hidden_weights;
output_weights_type::mat_type output_weights;

double hidden_weights_arr[hidden_weights_type::num_cells];
double output_weights_arr[output_weights_type::num_cells];

#ifdef SIMULATOR
    const char *hidden_weights_filename = "/tmp/in_hidden.csv";
    const char *output_weights_filename = "/tmp/hidden_out.csv";
#else
    #define EX3 "/data/private/robot/workspace/src/exercise3"
    #define WEIGHTS_BASE EX3 "/src/ANN_inv_sensor/early_stopping/output_weights"
    const char *hidden_weights_filename =
        WEIGHTS_BASE "/hidden_100/in_hidden_weights_30000.csv";
    const char *output_weights_filename =
        WEIGHTS_BASE "/hidden_100/hidden_out_weights_30000.csv";
#endif

// for each cell: the offset of the start of the block of 4 laser readings to
// take for that cell
unsigned cell_laser_offset[l_grid_y*l_grid_x];


// locations of the centers of each cell
Pol cell_locations[l_grid_y*l_grid_x]; // row major


static void read_csv(std::stringstream &data, const unsigned max_len, double *out) {
    unsigned i = 0;
    std::string token;
    while(std::getline(data, token, ',')) {
        boost::trim(token);
        if(token.length() == 0) // because of trailing , at the end of lines
            continue;

        assert(i < max_len);
        out[i] = atof(token.c_str());
        ++i;
    }
    assert(i == max_len);
}

void load_weights(const char *hidden_filename, const char *output_filename) {
    std::stringstream buffer;

    std::ifstream f1(hidden_filename);
    if(f1.is_open()) {
        buffer << f1.rdbuf();
        f1.close();

        read_csv(buffer, hidden_weights_type::num_cells, hidden_weights_arr);
        hidden_weights = Eigen::Map<hidden_weights_type::mat_type>(hidden_weights_arr);

        buffer.str(std::string()); // remove content
        buffer.clear(); // reset flags, not content

        ROS_INFO("loaded hidden weights from %s", hidden_filename);
    } else {
        ROS_ERROR("could not weights from file %s", hidden_filename);
    }

    std::ifstream f2(output_filename);
    if(f2.is_open()) {
        buffer << f2.rdbuf();
        f2.close();

        read_csv(buffer, output_weights_type::num_cells, output_weights_arr);
        output_weights = Eigen::Map<output_weights_type::mat_type>(output_weights_arr);

        ROS_INFO("loaded output weights from %s", output_filename);
    } else {
        ROS_ERROR("could not weights from file %s", output_filename);
    }
}


void output_precomputed() {
    std::ofstream out("precomp.py");

    assert(out.is_open());

    out << "#!/usr/bin/env python\n";

    out << "cell_locations = [";
    for(unsigned i = 0; i < l_grid_y*l_grid_x; ++i) {
        Pol loc = cell_locations[i];
        out << "(" << loc.r << "," << loc.theta << "), ";
        if((i+1) % l_grid_x == 0) out << "\n";
    }
    out << "]\n\n";

    out << "cell_laser_angles = [";
    for(unsigned i = 0; i < l_grid_y*l_grid_x; ++i) {
        unsigned l = cell_laser_offset[i];
        out << "("
            << laser_angle(l)   << ","
            << laser_angle(l+1) << ","
            << laser_angle(l+2) << ","
            << laser_angle(l+3)
            << "), ";
        if((i+1) % l_grid_x == 0) out << "\n";
    }
    out << "]\n\n";

    out.close();
}




void init_ANN() {

#if OCCUPANCY == OCC_ANN
    // in case no weights are available, at least initialise them to junk
    hidden_weights = Eigen::Map<hidden_weights_type::mat_type>(hidden_weights_arr);
    output_weights = Eigen::Map<output_weights_type::mat_type>(output_weights_arr);

    load_weights(hidden_weights_filename, output_weights_filename);
#endif


    // calculate grid cell locations
    assert(l_grid_x % 2 == 0 &&
            "grid should have even width so the robot sits between two cells");

    assert(readings_per_cell % 2 == 0 &&
            "must take an even number of readings per cell");

    double laser_angles[num_lasers];
    for(unsigned i = 0; i < num_lasers; ++i) {
        laser_angles[i] = laser_angle(i);
    }

    // position of the robot in grid cell coordinates (centers)
    /* Example: X denotes a grid cell and R denotes the robot grid cell
       XXXXXX
       XXXXXX
          R

       l_grid_x = 6,  l_grid_y = 2
       the robot is positioned at (3, 2). Note the cells are 0-indexed.
       This means the robot has a pseudo-cell below the actual grid.
       The top-left of R is used as the origin of the cell coordinate system
    */
    const int rob_x = l_grid_x / 2;
    const int rob_y = l_grid_y;

    for(unsigned y = 0; y < l_grid_y; ++y) {
        for(unsigned x = 0; x < l_grid_x; ++x) {
            // cell centers
            Pol loc = distance_from_cell_top_left(rob_x, rob_y, x, y).to_polar();
            cell_locations[y * l_grid_x + x] = loc;
            //ROS_INFO("(%i, %i) cells --> (%f, %f) metres", x, y, cell_x, cell_y);

            // scan through
            const unsigned width = readings_per_cell/2;
            const unsigned left_edge = num_lasers - width;
            for(unsigned i = 0; i < num_lasers; ++i) {
                if(laser_angles[i] > loc.theta) {
                    int offset;
                    if(i < width)
                        offset = 0;
                    else if(i >= left_edge)
                        offset = num_lasers-readings_per_cell;
                    else
                        offset = i - width;
                    assert(offset >= 0 && offset <= int(num_lasers)-int(readings_per_cell)
                            && "laser offset out of range");
                    cell_laser_offset[y * l_grid_x + x] = offset;
                    break;
                }
            }
        }
    }

    // TODO: precompute which cells <= max_range

    //output_precomputed();

}

// the input to the neural network is 4 laser readings (dependent on which cell)
// followed by the polar vector of the cell being tested. This function extracts
// the relevant data for a given cell
static void getInput(double *output, const float *lasers, const unsigned row, const unsigned col) {
    unsigned cell_index = row * l_grid_x + col;
    Pol *p = &cell_locations[cell_index];
    unsigned l = cell_laser_offset[cell_index]; // start index of the laser readings to take
    output[0] = lasers[l];
    output[1] = lasers[l+1];
    output[2] = lasers[l+2];
    output[3] = lasers[l+3];
    output[4] = p->theta; // why this order, not (r, theta)? Ask Charlie...
    output[5] = p->r;
}

inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


// calculate an occupancy grid using an artificial neural network
// caller should hold laser_mutex
void ANN_occupancy(const sensor_msgs::LaserScan::ConstPtr &last_laser_data) {

#ifdef SIMULATOR
    load_weights(hidden_weights_filename, output_weights_filename);
#endif

    float32 *map = &local_map.local_map[0];

    assert(last_laser_data->ranges.size() == num_lasers &&
            "ANN tuned to 512 readings only");

    // modify laser data in place (hope ROS doesn't mind)
    // filter out NaN and out of range values the same as sense_and_drive did
    {
        float *lasers = const_cast<float*>(&last_laser_data->ranges[0]);
        const float min = last_laser_data->range_min;
        const float max = last_laser_data->range_max;

        for(unsigned i = 0; i < num_lasers; ++i) {
            float laser = lasers[i];
            if(isnan(laser) || laser < min)
                lasers[i] = min;
            else if(laser > max)
                lasers[i] = max;
        }
    }


    //float *inputs;
    // 4 sensor readings
    // 1 angle
    // 1 distance

    Mat<input_dimensions, 1>::single_row_mat_type input;
    Mat<hidden_neurons, 1>::single_row_mat_type signals_to_hidden;
    double *data = signals_to_hidden.data();
    Mat<1, 1>::single_row_mat_type output_mat;
    const float *ranges = &last_laser_data->ranges[0];

    for(unsigned row = 0; row < l_grid_y; ++row) {
        for(unsigned col = 0; col < l_grid_x; ++col) {
            getInput(input.data(), ranges, row, col);

            signals_to_hidden = hidden_weights * input;

            for(unsigned i = 0; i < hidden_neurons * 1; ++i)
                data[i] = tanh(data[i]);

            output_mat = output_weights * signals_to_hidden;
            map[row * l_grid_x + col] = float(sigmoid(output_mat(0, 0)));
            //std::cout << map[row * l_grid_x + col] << std::endl;
        }
    }

}







// calculate a random occupancy grid
// caller should hold laser_mutex
void random_occupancy() {
    float32 *cur = &local_map.local_map[0]; // cursor
    for(unsigned i = 0; i < l_grid_x*l_grid_y; ++i) {
        *cur++ = rand()/(float32)RAND_MAX;
    }
}






// get the index of a cell given by a polar coordinate. return -1 if out of range
int cell_index(Pol p) {
    const int rob_x = l_grid_x / 2;

    const Point lp = p.to_point();

    // cell indexing starts from the top left of the grid
    // determine which grid cell the laser reading falls into
    // floor towards -infinity eg floor(-2.3) == -3.0
    const int tl_x = int(floor(lp.x / grid_resolution));
    const int tl_y = int(l_grid_y) - int(ceil(lp.y / grid_resolution));

    // determine the grid cell, with the origin at the top left of the grid,
    // from the coordinates (tl_x, tl_y) which are the top left coordinates
    // of the cell in question with (rob_x, rob_y) as the origin

    const int cell_x = tl_x + rob_x;
    const int cell_y = tl_y;

    //ROS_INFO("%i: (%i,%i)cell = (%i,%i)tl = (%f,%f)m = (%f,%f)pol",
             //i, cell_x, cell_y, tl_x, tl_y lp.x, lp.y, l, angle);

    if(cell_x >= 0 && cell_x < int(l_grid_x) &&
       cell_y >= 0 && cell_y < int(l_grid_y)) {
        return cell_y*l_grid_x + cell_x;
    } else {
        return -1;
    }
}

void place_wall(float l, float angle, float32 *map, float wall_certainty) {
    int index = cell_index(Pol(l+grid_resolution, angle));
    if(index != -1) {
        if(map[index] < 0) // cell was free
            map[index] = wall_certainty; // reset to 0 and add the wall
        else
            map[index] += wall_certainty;
    }
}

bool true_with_prob(float p) {
    return float(rand())/RAND_MAX > p;
}

// calculate an occupancy grid by a simple AABB test
// caller should hold laser_mutex
void AABB_occupancy(const sensor_msgs::LaserScan::ConstPtr &last_laser_data) {
    float32 *map = &local_map.local_map[0];
    memset(map, 0, sizeof(float32)*l_grid_x*l_grid_y);

    // documentation says to throw away ranges not in [min,max]
    const float32 minr = std::max(double(last_laser_data->range_min), 0.0001);
    const float32 maxr = last_laser_data->range_max;

    const float *lasers = &last_laser_data->ranges[0];

    const unsigned len = last_laser_data->ranges.size();
    assert(len == 512 && "laser_angle() is tuned to 512 readings only");

    // EXPLANATION
    // negative values are used to signify the likelihood of being free
    // positive values are used to signify the likelihood of being occupied
    // 0 are untouched cells
    // at the end of this function: these values are transformed into probabilities


    for(unsigned i = 0; i < len; i += 1) {
        float32 l = lasers[i];
        if(l > maxr)
            l = maxr;
        if(isnan(l) || l < minr)
            continue;


        const double angle = laser_angle(i);

        // lay a wall down if the laser is not at its maximum (ie it stopped
        // because it hit something)
        if(l != maxr) {
            place_wall(l, angle, map, 0.8);
            if(true_with_prob(0.6)) {
                place_wall(l-1*grid_resolution, angle, map, 0.8);
                place_wall(l-2*grid_resolution, angle, map, 0.7);
                if(true_with_prob(0.8)) {
                    place_wall(l-3*grid_resolution, angle, map, 0.7);
                    place_wall(l-4*grid_resolution, angle, map, 0.7);
                }
            }
        }

        l = l-grid_resolution;
        while(l > 0) {
            int index = cell_index(Pol(l, angle));

            if(index != -1)
                if(map[index] <= 0) // not occupied
                    map[index] -= 1;

            l -= grid_resolution;
        }

    }

    // now the local map contains a count of the number of laser readings which
    // fall into each cell.
    // apply a function to these counts to make them tend to 1 rather than infinity

    // k determines the gradient at x=0. Larger => approach 1 faster
    const float32 kfree = 0.25;
    const float32 kocc = 0.25;
    for(unsigned i = 0; i < l_grid_x*l_grid_y; ++i) {
        if(map[i] == 0) // no lasers hit this cell
            map[i] = 0.5;
        else if(map[i] < 0) // free
            // negate the value on the map to get a positive value
            map[i] = exp(-kfree*-map[i]);
        else // occupied
            map[i] = 1.0 - exp(-kocc*map[i]);

        if(map[i] < 0.05) map[i] = 0.05;
        if(map[i] > 0.9) map[i] = 0.9;
    }
}



// standard inverse sensor technique described in probabilistic robotics chapter 9
void std_occupancy(const sensor_msgs::LaserScan::ConstPtr &last_laser_data) {
    float32 *map = &local_map.local_map[0];

    const float *lasers = &last_laser_data->ranges[0];
    const float32 maxr = last_laser_data->range_max;

#ifdef SIMULATOR
    float alpha = 0.2; // metres
#else
    float alpha = 0.2; // metres
#endif

    for(unsigned i = 0; i < l_grid_y*l_grid_x; ++i) {
        int laser_offset = cell_laser_offset[i];
        Pol cell_loc = cell_locations[i]; // cell centre in metres
        if(cell_loc.r < maxr - grid_resolution) {
            float l1 = lasers[laser_offset+0];
            float l2 = lasers[laser_offset+1];
            float l3 = lasers[laser_offset+2];
            float l4 = lasers[laser_offset+3];


            if((l1 < maxr && fabs(l1-cell_loc.r) < alpha) ||
               (l2 < maxr && fabs(l2-cell_loc.r) < alpha) ||
               (l3 < maxr && fabs(l3-cell_loc.r) < alpha) ||
               (l4 < maxr && fabs(l4-cell_loc.r) < alpha))
                map[i] = 0.95; // probably wall
            else if(l2 >= cell_loc.r || l3 >= cell_loc.r)
                map[i] = 0.05; // probably not blocked
            else
                map[i] = 0.5; // behind wall, not sure
        } else {
            map[i] = -1; // outside range => padding
        }
    }
}

