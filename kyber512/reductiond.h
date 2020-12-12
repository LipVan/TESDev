/*
 * reductiond.h
 *
 *  Created on: Dec 4, 2020
 *      Author: lip
 */

#ifndef REDUCTIOND_H_
#define REDUCTIOND_H_

#include <stdint.h>
#include "params.h"

__device__ int16_t montRdc16Dev(int32_t in);

__device__ int16_t barrtRdc16Dev(int16_t in);

#endif /* REDUCTIOND_H_ */
