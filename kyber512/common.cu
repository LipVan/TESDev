#include "common.h"
using namespace nvcuda;


/*
 * CBD: Convert 128 bytes to 256 coefficients
 */

__device__ void CBDEtaDev(int8_t *out_poly, uint8_t *in_B)
{
	uint8_t tmp;
	int8_t a;
	int8_t b;

	for(int i=0; i<64*KYBER_ETA; i++)
	{
		for(int j=0; j<KYBER_ETA; j++)
		{
			tmp = (in_B[i] >> (1-j)*4) & 0xf;
			a = (tmp & 0x8)>>3 + (tmp & 0x4)>>2;
			b = (tmp & 0x2)>>1 + (tmp & 0x1);

			out_poly[i*KYBER_ETA+j] = (int8_t)(a-b);
		}
	}
}
/*
 * Parse: B* -> Rqn
 * Uniform sampling in Rq
 * */
__device__ int ParseDev(int16_t *out_poly, uint req_len, uint8_t *bytes, uint byte_len)
{
	uint16_t d;
	int i=0;
	int ctr=0;

	while(ctr < req_len && i+2<= byte_len)
	{
		d = ((uint16_t)bytes[i+1] << 8) | bytes[i];
		if(d < NINETEEN_Q)
		{
			d -= (d >> 12) * KYBER_Q;
			out_poly[ctr++] = (int16_t)d;
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

__device__ void GensreDev(int8_t *vec, uint8_t *seed, uint8_t nounce)
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


//Signed char ->16x16x16
//__global__ void nttVecI8Ker(int16_t *o_vec, int8_t *i_vec, int8_t *i_tabg)
//{
//	//Perhaps the shared memory is not enough for KYBER_N*KYBER_N = 64 kB
////	__shared__ int8_t shmem[KYBER_N*KYBER_N];
//
//	//Copy data from global memory to shared memory
//
//	int warp_id = threadIdx.x / WARP_SIZE;
//
//	wmma::fragment<wmma::matrix_a, WMM_M, WMM_N, WMM_K, char, wmma::row_major> a_frag;
//	wmma::fragment<wmma::matrix_b, WMM_M, WMM_N, WMM_K, char, wmma::row_major> b_frag;
//	wmma::fragment<wmma::accumulator, WMM_M, WMM_N, WMM_K, int> c_frag[4];
//
//	for(int i=0; i<4; ++i)
//	{
//		wmma::fill_fragment(c_frag[i], 0);
//	}
//
//	//c0 = a0 * b0
//	for(int round=0; round<KYBER_N/WMM_N; ++round)
//	{
//		int8_t *a_warp_ptr = (int8_t *)i_vec + round*WMM_N;
//
//		wmma::load_matrix_sync(a_frag, a_warp_ptr, KYBER_N);
//
//#pragma unroll
//		for(int ind=0; ind<4; ++ind)
//		{
//			int8_t *b_warp_ptr = (warp_id < WARPS_PER_BLOCK/2)? (i_tabg + round*WMM_N*KYBER_N + (warp_id*2 + ind)*WMM_K):
//																(i_tabg + TABLE_OFFSET + round*WMM_N*KYBER_N + (warp_id*2 + ind)*WMM_K);
//
//			wmma::load_matrix_sync(b_frag, b_warp_ptr , KYBER_N);
//			wmma::mma_sync(c_frag[ind], a_frag, b_frag, c_frag[ind]);
//		}
//	}
//}

__global__ void nttVecI8Ker(int32_t *o_vec, int8_t *i_vec, int8_t *i_tabg)
{
	//Perhaps the shared memory is not enough for KYBER_N*KYBER_N = 64 kB
//	__shared__ int8_t shmem[KYBER_N*KYBER_N];

	//Copy data from global memory to shared memory

	int warp_id = threadIdx.x / WARP_SIZE;

	wmma::fragment<wmma::matrix_a, WMM_M, WMM_N, WMM_K, char, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMM_M, WMM_N, WMM_K, char, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, WMM_M, WMM_N, WMM_K, int> c_frag[2];

	for(int i=0; i<2; ++i)
	{
		wmma::fill_fragment(c_frag[i], 0);
	}

	//c0 = a0 * b0
	for(int round=0; round<KYBER_N/WMM_N; ++round)
	{
		int8_t *a_warp_ptr = (int8_t *)i_vec + round*WMM_N;

		wmma::load_matrix_sync(a_frag, a_warp_ptr, KYBER_N);

#pragma unroll
		for(int ind=0; ind<2; ++ind)
		{
			int8_t *b_warp_ptr = i_tabg + round*WMM_N*KYBER_N + (warp_id*2 + ind)*WMM_K;

			wmma::load_matrix_sync(b_frag, b_warp_ptr , KYBER_N);
			wmma::mma_sync(c_frag[ind], a_frag, b_frag, c_frag[ind]);
		}
	}

	//c0 <-- c0*BASE_BITS
	for(int ind=0; ind<2; ++ind)
	{
		for(int t=0; t<c_frag[ind].num_elements; ++t)
		{
			c_frag[ind].x[t] *= BASE_BITS;
		}
	}

	//c1 <- a1 * b1 + c0
	for(int round=0; round<KYBER_N/WMM_N; ++round)
	{
		int8_t *a_warp_ptr = (int8_t *)i_vec + round*WMM_N;
		wmma::load_matrix_sync(a_frag, a_warp_ptr, KYBER_N);

#pragma unroll
		for(int ind=0; ind<2; ++ind)
		{
			int8_t *b_warp_ptr = (int8_t *)i_tabg + TABLE_OFFSET + round*WMM_N*KYBER_N + (warp_id*2 + ind)*WMM_K;

			wmma::load_matrix_sync(b_frag, b_warp_ptr, KYBER_N);
			wmma::mma_sync(c_frag[ind], a_frag, b_frag, c_frag[ind]);
		}
	}

	//Store the result
	for(int ind=0; ind<2; ++ind)
	{
		int *o_warp_ptr = o_vec + (warp_id*2 + ind)*WMM_K;

		wmma::store_matrix_sync(o_warp_ptr, c_frag[ind], KYBER_N);
	}
//	__syncthreads();
}
