
#ifndef UPDATE_H
#define UPDATE_H

#include "stitching.h"

void update_global_map(FittedMap, OccGrid*);
void log_odds_update_global_map(FittedMap, OccGrid*, OccGrid*);
void fake_update_global_map(FittedMap, OccGrid*);

#endif
