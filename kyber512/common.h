/*
 * common.h
 *
 *  Created on: Oct 15, 2020
 *      Author: lip
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "keccak.h"
#include "params.h"
#include "rng.h"

#define TEST_CASES	10240

#define BLOCK_SIZE 	256
#define WARP_SIZE	32

__device__ void CBDEtaDev(int16_t *out_poly, uint8_t *in_B);

__device__ int ParseDev(int16_t *out_poly, uint req_len, uint8_t *bytes, uint byte_len);

__device__ void GenMatrixADev(int16_t *matrixA, uint8_t *rho);

__device__ void GensreDev(int16_t *vec, uint8_t *seed, uint8_t nounce);

#endif /* COMMON_H_ */
