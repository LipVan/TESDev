/*
 * main.cpp
 *
 *  Created on: Nov 27, 2020
 *      Author: lip
 */

#include <iostream>
#include <fstream>

using namespace std;

#define KYBER_N 256
#define KYBER_Q 3329
#define ZETAS   17
#define MATRIX_OFFSET   (KYBER_N*KYBER_N)

int main()
{
    uint16_t result[2][KYBER_N*KYBER_N]={{0}};

    for (int i = 0; i < 2; ++i)
    {
        for (int row = 0; row < KYBER_N; ++row)
        {
            for (int col = 0; col < KYBER_N; ++col)
            {
                result[i][row*KYBER_N + col] = 1;
            }
        }
    }

    uint16_t tmp_g = 1;
    uint16_t tmp_res = 1;

    for (int col = 1; col < KYBER_N; ++col)
    {
        tmp_g = (tmp_g * ZETAS) % KYBER_Q;
        tmp_res = 1;

        for (int i = 1; i < KYBER_N; ++i)
        {
            tmp_res = (tmp_res * tmp_g) % KYBER_Q;

            //result[1] stores the low 7bit, and result[0] for rest high.
            result[1][i*KYBER_N + col] = tmp_res & 0x7f;
            result[0][i*KYBER_N + col] = (tmp_res &0x7fff) >> 7;
        }
        cout<<tmp_res<<",";
    }

    ofstream out("GTab");
    if(!out.is_open())
    {
        cout<<"Error to ctreate output file!"<<endl;
        return -1;
    }

    out<<"uint8_t GTab[2][KYBER_N*KYBER_N] = {{";
    for (int i = 0; i < KYBER_N; ++i)
    {
    	for(int j=0; j<KYBER_N; ++j)
    	{
    		out<<result[0][i*KYBER_N+j]<<",";
    	}
    	out<<"\\"<<endl;
    }
    out<<"},{";

    for (int i = 0; i < KYBER_N; ++i)
    {
    	for(int j=0; j<KYBER_N; ++j)
    	{
    		out<<result[1][i*KYBER_N+j]<<",";
    	}
    	out<<"\\"<<endl;
    }
    out<<"}}";

    out.close();

    cout<<"Finished !"<<endl;
    return 0;


}


