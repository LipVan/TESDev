/*
 * zeta_table.h
 *
 *  Created on: Nov 23, 2020
 *      Author: lip
 */

#ifndef KYBER512_ZETA_TABLE_H_
#define KYBER512_ZETA_TABLE_H_
#include <stdint.h>
#include "common.h"

extern __device__ int16_t zetaTab[];

extern __device__ int8_t GTab[2*KYBER_N*KYBER_N];
#endif /* KYBER512_ZETA_TABLE_H_ */
