# header
uint32 counter
time   timestamp

# change in pose since last local map
float64 dx
float64 dy
float64 dtheta

# occupancy grid
float32   resolution # m/cell
uint32    cols       # aka stride
float32[] local_map
