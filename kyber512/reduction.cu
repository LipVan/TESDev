/*
 * reduction.cu
 *
 *  Created on: Dec 4, 2020
 *      Author: lip
 */

#include "reductiond.h"

//Compute 16--bit congruent to _in_ * R^-1 mod q, where R=2^16
__device__ int16_t montRdc16Dev(int32_t in)
{
	int32_t tmp;
	int16_t u;

	u = in * KYBER_Q_INV;
	tmp = (int32_t) u *KYBER_Q;
	tmp = in - tmp;
	tmp >>= 16;

	return tmp;
}

#define BARRET_REDC_V	((1U << 26)/KYBER_Q + 1)

//Compute 16-bit integer congruent to _in_ mod q in {0,...,q}
__device__ int16_t barrtRdc16Dev(int16_t in)
{
	int32_t tmp = BARRET_REDC_V * in;

	tmp >>= 26;
	tmp *= KYBER_Q;

	return tmp;
}
