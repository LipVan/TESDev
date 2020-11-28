#include "keygen.h"



__global__ void keyGenKer(int16_t *t, int16_t *matrix_A, int8_t *s, int8_t *e, uint8_t *seed)
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
		GensreDev(s + tid *KYBER_K*KYBER_N, rho_sigma+32, nounce);
		++nounce;
	}

	//Gen vector e
	for(int i=0; i<KYBER_K; ++i)
	{
		GensreDev(e + tid *KYBER_K*KYBER_N, rho_sigma+32, nounce);
		++nounce;
	}
	__syncthreads();

	//Raw NTT for the vectors
	nttVecI8Ker()

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
