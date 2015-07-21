/*
 * viterbi_detector.h
 *
 *  Created on: 20-Jul-2015
 *      Author: akib
 */

#ifndef VITERBI_DETECTOR_H_
#define VITERBI_DETECTOR_H_



void viterbi_detector(const gr_complex * input, unsigned int samples_num, gr_complex * rhh, unsigned int start_state, const unsigned int * stop_states, unsigned int stops_num, float * output);


#endif /* VITERBI_DETECTOR_H_ */
