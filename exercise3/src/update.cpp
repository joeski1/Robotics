
#include "update.h"
#include "stitching.h"


const float PADDING = -1;

void update_global_map(FittedMap localMap, OccGrid *globalMap) {
    // location of the top left of the local map on the global map (in cells)
    int local_map_x = globalMap->robot_x - localMap.map->robot_x;
    int local_map_y = globalMap->robot_y - localMap.map->robot_y;

	//assert that the localmap fits within the global map
	assert(local_map_x >= 0 && local_map_x + localMap.map->width  < globalMap->width &&
           local_map_y >= 0 && local_map_y + localMap.map->height < globalMap->height);


	for(unsigned y = 0; y < localMap.map->height; y++) {
		for(unsigned x = 0; x < localMap.map->width; x++) {
			//get the localmap map probability value
			float Pl = localMap.map->getCell(x, y);

			// ignore padding cells
			if (Pl != PADDING) {
                //assert(0.0 <= Pl && Pl <= 1);

                unsigned cell_x = local_map_x + x;
                unsigned cell_y = local_map_y + y;

				//get the global map probability value
				float Pg = globalMap->getCell(cell_x, cell_y);
                //assert(0.0 <= Pg && Pg <= 1);

				//update the global grid
				//float tmp  = 1/(1-P) - 1;
				//float newP = 1 - (1 / (1 + tmp * Pn));
                //float newP = 1 - pow(1 + (pow(1-P,-1) - 1) * Pn,-1)

				float tmp = (Pl/(1.0-Pl))*(Pg/(1.0-Pg));
				float newP = 1.0 - 1.0/(1.0 + tmp);

				if(newP < 0.05) newP = 0.05;
				if(newP > 0.95) newP = 0.95;

				globalMap->setCell(cell_x, cell_y, newP);
			}
		}
	}
}

void log_odds_update_global_map(FittedMap localMap, OccGrid *logMap, OccGrid *probMap) {
    // location of the top left of the local map on the global map (in cells)
    int local_map_x = logMap->robot_x - localMap.map->robot_x;
    int local_map_y = logMap->robot_y - localMap.map->robot_y;

	//assert that the localmap fits within the global map
	assert(local_map_x >= 0 && local_map_x + localMap.map->width  < logMap->width &&
           local_map_y >= 0 && local_map_y + localMap.map->height < logMap->height);

    float too_certain_occ     = log(0.99/(1-0.99));
    float too_certain_not_occ = log(0.01/(1-0.01));

	for(unsigned y = 0; y < localMap.map->height; ++y) {
		for(unsigned x = 0; x < localMap.map->width; ++x) {

			// local map stored as probability
			float p = localMap.map->getCell(x, y);

			if(p != PADDING) {
                float logl = log(p/(1-p));

                unsigned cell_x = local_map_x + x;
                unsigned cell_y = local_map_y + y;


				float logg = logMap->getCell(cell_x, cell_y);
                // probabilistic robotics table 4.2 simplified with prior of 0.5
                float logp = logg + logl;

                // clamp
                if(logp > too_certain_occ)
                    logp = too_certain_occ;
                else if(logp < too_certain_not_occ)
                    logp = too_certain_not_occ;

                logMap->setCell(cell_x, cell_y, logp);

                // convert from log odds to probability
                probMap->setCell(cell_x, cell_y, 1 - 1 / (1 + exp(logp)));
            }
        }
    }
}

void fake_update_global_map(FittedMap localMap, OccGrid *globalMap) {
    // location of the top left of the local map on the global map (in cells)
    int local_map_x = globalMap->robot_x - localMap.map->robot_x;
    int local_map_y = globalMap->robot_y - localMap.map->robot_y;

	//assert that the localmap fits within the global map
	assert(local_map_x >= 0 && local_map_x + localMap.map->width  < globalMap->width &&
           local_map_y >= 0 && local_map_y + localMap.map->height < globalMap->height);

	for(unsigned y = 0; y < localMap.map->height; ++y) {
		for(unsigned x = 0; x < localMap.map->width; ++x) {
			//get the localmap map probability value
			float Pl = localMap.map->getCell(x, y);
			// ignore padding cells
			if (Pl != PADDING) {
                unsigned cell_x = local_map_x + x;
                unsigned cell_y = local_map_y + y;

				float global_weight = 5; // more weight to what is already present
				float Pg = globalMap->getCell(cell_x, cell_y);
                globalMap->setCell(cell_x, cell_y,
                        (Pl+Pg*global_weight)/(global_weight+1));
			}
		}
	}
}
