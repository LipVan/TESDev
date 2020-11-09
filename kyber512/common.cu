#include "common.h"

/*
 * CBD: Convert 128 bytes to 256 coefficients
 */

__device__ void CBDEtaDev(int16_t *out_poly, uint8_t *in_B)
{
	uint8_t tmp;
	uint8_t a;
	uint8_t b;

	for(int i=0; i<64*KYBER_ETA; i++)
	{
		for(int j=0; j<KYBER_ETA; j++)
		{
			tmp = (in_B[i] >> (1-j)*4) & 0xf;
			a = (tmp & 0x8)>>3 + (tmp & 0x4)>>2;
			b = (tmp & 0x2)>>1 + (tmp & 0x1);

			out_poly[i*KYBER_ETA+j] = (int16_t)(a-b);
		}
	}
}
/*
 * Parse: B* -> Rqn
 * Uniform sampling in Rq
 * */
__device__ int ParseDev(int16_t *out_poly, uint req_len, uint8_t *bytes, uint byte_len)
{
	int16_t d;
	int i=0;
	int ctr=0;

	while(ctr < req_len && i+2< byte_len)
	{
		d = (int16_t)((uint16_t)bytes[i+1] << 8) | bytes[i];
		if(d < NINETEEN_Q)
		{
			out_poly[ctr++] = d;
		}
		i += 2;
	}

	return ctr;
}


__device__ void GenMatrixADev(int16_t *matrixA, uint8_t *rho)
{
	uint64_t state[25]={0};
	uint8_t exseed[KYBER_SEED_LEN + 2];
	int ctr=0;
	uint8_t output[XOF_BLOCKBYTES];

	for(int i=0; i<KYBER_SEED_LEN; i++) exseed[i] = rho[i];

	for(int i=0; i<KYBER_K; i++)
	{
		for(int j=0; j<KYBER_K; j++)
		{
			int16_t *tmp_ptr = matrixA + (i*2+j)*KYBER_N;

			exseed[KYBER_SEED_LEN] = (uint8_t)j;
			exseed[KYBER_SEED_LEN+1] = (uint8_t)i;

			keccak1600AbsorbDev(state, SHAKE128_RATE, exseed, KYBER_SEED_LEN+2, 0x1F);

			keccak1600SqueezeDev(output, 1, state, SHAKE128_RATE);

			ctr = ParseDev(tmp_ptr, KYBER_N, output, XOF_BLOCKBYTES);

			while(ctr < KYBER_N)
			{
				keccak1600SqueezeDev(output, 1, state, SHAKE128_RATE);
				ctr += ParseDev(tmp_ptr+ctr, KYBER_N-ctr, output, XOF_BLOCKBYTES);
			}
		}
	}
}

__device__ void GensreDev(int16_t *vec, uint8_t *seed, uint8_t nounce)
{
	uint8_t exseed[KYBER_SEED_LEN+1];
	uint8_t output[KYBER_ETA*KYBER_N/4];

	for(int i=0; i<KYBER_SEED_LEN; i++) exseed[i] = seed[i];

	for(int i=0; i<KYBER_K; ++i)
	{
		exseed[KYBER_SEED_LEN] = i;
		uint64_t state[25]={0};

		keccak1600AbsorbDev(state, SHAKE256_RATE, exseed, KYBER_SEED_LEN+1, 0x1F);
		keccak1600SqueezeDev(output, KYBER_ETA*KYBER_N/4/XOF_BLOCKBYTES, state, SHAKE256_RATE);

		CBDEtaDev(vec, output);
	}
}
