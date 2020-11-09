/*
 * fips202.h
 *
 *  Created on: Oct 29, 2020
 *      Author: lip
 */

#ifndef FIPS202_H_
#define FIPS202_H_

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE  72
#define XOF_BLOCKBYTES 168

__global__ void shake128Dev(uint8_t *output, uint64_t out_byte_len, uint64_t *s, const uint8_t *input, uint32_t input_byte_len);

__device__ void keccak1600PermuteDev(uint64_t *state);

__device__ void keccak1600AbsorbDev(uint64_t *state, uint r, uint8_t *m, uint64_t mlen, uint8_t suffix);

__device__ void keccak1600SqueezeDev(uint8_t *output, uint nblocks, uint64_t *state, uint r);

__device__ void SHA3_512Dev(uint8_t *out, uint8_t *in, uint16_t in_len);

__device__ void SHAKE128_Dev(uint8_t *out, uint out_nblocks, uint8_t *in, uint in_len);


#endif /* FIPS202_H_ */
