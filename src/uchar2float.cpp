/*
 * uchar2float.cpp
 *
 *  Created on: 18-Jul-2015
 *      Author: akib
 */


#include<uchar2float.h>

float *
uchar_array_to_float (const unsigned char *in, int nsamples)
{
	float out[FLOAT_ARRAY_SIZE];
	int i ;
	for(i=0;i<nsamples;i++)
	{
		out[i]=(float)(in[i]);
	}
	return out;
}
