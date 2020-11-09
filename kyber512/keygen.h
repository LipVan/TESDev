/*
 * keygen.h
 *
 *  Created on: Oct 15, 2020
 *      Author: lip
 */

#ifndef KEYGEN_H_
#define KEYGEN_H_

#include "common.h"

__global__ void keyGenKer();

__global__ void keyGenKer(int16_t *matrix_A, int16_t *s, uint8_t *seed)

void keyGenCKer(uint8_t *pk, uint8_t *sk);


#endif /* KEYGEN_H_ */
