/*
 * params.h
 *
 *  Created on: Nov 5, 2020
 *      Author: lip
 */

#ifndef KYBER512_PARAMS_H_
#define KYBER512_PARAMS_H_

#define KYBER_512


#define KYBER_N			256
#define KYBER_Q			3329
#define KYBER_Q_INV		62209	 	//q^(-1) mod 2^16
#define NINETEEN_Q	(19 * KYBER_Q)
#define KYBER_ETA		2

#define KYBER_SEED_LEN	32

#ifdef KYBER_512
#define KYBER_K		2
#define KYBER_DU	10
#define KYBER_DV	3
#define KYBER_SK_LEN	1632
#define KYBER_PK_LEN	800
#define KYBER_CT_LEN	736
#endif

#ifdef KYBER_768
#define KYBER_K		3
#define KYBER_DU	10
#define KYBER_DV	4
#define KYBER_SK_LEN	2400
#define KYBER_PK_LEN	1184
#define KYBER_CT_LEN	1088
#endif

#ifdef KYBER_1024
#define KYBER_K		4
#define KYBER_DU	11
#define KYBER_DV	5
#define KYBER_SK_LEN	3168
#define KYBER_PK_LEN	1568
#define KYBER_CT_LEN	1568
#endif


#endif /* KYBER512_PARAMS_H_ */
