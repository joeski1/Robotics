# header
uint32 counter
time timestamp

# bake the current pose estimate into the global map message for a consistent
# view of the data.
exercise3/pose_update pose

# occupancy grid
float32 resolution
uint32 cols
float32[] global_map

