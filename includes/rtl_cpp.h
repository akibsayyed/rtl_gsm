/*
 * rtl_cpp.h
 *
 *  Created on: 17-Jul-2015
 *      Author: akib
 */

#ifndef RTL_CPP_H_
#define RTL_CPP_H_
#include<gnuradio/gr_complex.h>
#include<stdio.h>
#include <string.h>
#include <stdlib.h>
#include<uchar2float.h>
using namespace std;

//std::vector<gr_complex> _lut;
unsigned short *_buf;
FILE *file ,*file2,*file3;
gr_complex *out;

#endif /* RTL_CPP_H_ */
