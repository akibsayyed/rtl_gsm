/*
 * uchar2float.h
 *
 *  Created on: 18-Jul-2015
 *      Author: akib
 */

#ifndef UCHAR2FLOAT_H_
#define UCHAR2FLOAT_H_


#define FLOAT_ARRAY_SIZE (16 * 16384)/2
float * uchar_array_to_float (const unsigned char *in, int nsamples);
#endif /* UCHAR2FLOAT_H_ */
