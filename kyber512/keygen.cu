#include "keygen.h"

//FP32 -> 16x16x8
//Block_size = 256 -> warps = 8
//(s,e) -> (^s, ^e), 8 tuples per warp
__global__ void nttVecKer(float *out_d, float *in_a, float *in_b, float *in_c)
{
	__shared__ float shmem[(KYBER_N/4)*KYBER_N];
	int warp_id = threadIdx.x / WARP_SIZE;

	//4 Rounds
#pragma unroll
	for(int round=0; round<4; round++)
	{
		//Copy 1/4 Zeta matrix to shared memory
		for(int i=0; i<)
	}
}

__global__ void keyGenKer(int16_t *t, int16_t *matrix_A, int16_t *s, int16_t *e, uint8_t *seed)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	uint8_t rho_sigma[64];

	//hash_G
	SHA3_512Dev(rho_sigma, seed + KYBER_SEED_LEN * tid, KYBER_SEED_LEN);

	//Gen matrix A
	GenMatrixADev((int16_t *)(matrix_A + tid*(KYBER_K*KYBER_K*KYBER_N)), (uint8_t *)rho_sigma);

	//Gen vector s
	uint8_t nounce=0;
	for(int i=0; i<KYBER_K; ++i)
	{
		GensreDev(s+tid*KYBER_K*KYBER_N, rho_sigma+32, nounce);
		++nounce;
	}

	//Gen vector e
	for(int i=0; i<KYBER_K; ++i)
	{
		GensreDev(e+tid*KYBER_K*KYBER_N, rho_sigma+32, nounce);
		++nounce;
	}
	__syncthreads();

	//Raw NTT for the vectors


}

void keyGenCKer(uint8_t *pk, uint8_t *sk)
{

}

void testKeyGenCKer()
{
	uint8_t *h_seed = (uint8_t *)malloc(KYBER_SEED_LEN * TEST_CASES);


	uint8_t *d_seed;
	uint8_t *d_matrix_A;
	uint8_t *d_s;

	cudaMalloc(&d_seed, KYBER_SEED_LEN * TEST_CASES);
	cudaMalloc(&d_matrix_A, KYBER_N * KYBER_K * KYBER_K * TEST_CASES);
	cudaMalloc(&d_s, KYBER_N * KYBER_K * TEST_CASES);

	//Generate seed d
	randombytes(h_seed, KYBER_SEED_LEN*TEST_CASES);
	cudaMemcpy(d_seed, h_seed, KYBER_SEED_LEN*TEST_CASES, cudaMemcpyHostToDevice);

	//Call Kernel
	keyGenKer(d_matrix_A, d_s, d_seed);

	free(h_seed);
	cudaFree(d_seed);
	cudaFree(d_matrix_A);
	cudaFree(d_s);
}
