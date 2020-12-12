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
#include <mma.h>

#include "keccak.h"
#include "params.h"
#include "rng.h"
#include "zeta_table.h"

#define GRID_SIZE	(46*2)
#define BLOCK_SIZE 	256

#define TEST_ROUNDS	10
#define TEST_CASES	(TEST_ROUNDS * GRID_SIZE * BLOCK_SIZE)

#define WARP_SIZE	32
#define WARPS_PER_BLOCK	(BLOCK_SIZE/WARP_SIZE)

#define WMM_M	16
#define WMM_N	16
#define WMM_K	16

#define SKEW_CHAR 		(256/8)
#define LINE_STRIDE		(KYBER_N + SKEW_CHAR)

#define BASE_BITS	(2^7)
#define TABLE_OFFSET	(KYBER_N*KYBER_N)

__device__ void CBDEtaDev(int16_t *out_poly, uint8_t *in_B);

__device__ int ParseDev(int16_t *out_poly, uint req_len, uint8_t *bytes, uint byte_len);

__device__ void GenMatrixADev(int16_t *matrixA, uint8_t *rho);

__device__ void GensreDev(int16_t *vec, uint8_t *seed, uint8_t nounce);

__global__ void nttVecI8Ker(int32_t *o_vec, int8_t *i_vec, int8_t *i_tab);

#endif /* COMMON_H_ */
