/*
 * main.h
 *
 *  Created on: 20-Jul-2015
 *      Author: akib
 */

#ifndef MAIN_H_
#define MAIN_H_
#include<sch.h>
void gsm_decode(gr_complex* input,int noutput_items);
        /** Function whis is used to search a FCCH burst and to compute frequency offset before
        * "synchronized" state of the receiver
        *
        * @param input vector with input signal
        * @param nitems number of samples in the input vector
        * @return
        */
        bool find_fcch_burst(const gr_complex *input, const int nitems, double & computed_freq_offset);

        /** Computes frequency offset from FCCH burst samples
         *
         * @param[in] input vector with input samples
         * @param[in] first_sample number of the first sample of the FCCH busrt
         * @param[in] last_sample number of the last sample of the FCCH busrt
         * @param[out] computed_freq_offset contains frequency offset estimate if FCCH burst was located
         * @return true if frequency offset was faound
         */
        double compute_freq_offset(const gr_complex * input, unsigned first_sample, unsigned last_sample);

        /** Computes angle between two complex numbers
         *
         * @param val1 first complex number
         * @param val2 second complex number
         * @return
         */
        inline float compute_phase_diff(gr_complex val1, gr_complex val2);

        /** Function whis is used to get near to SCH burst
         *
         * @param nitems number of samples in the gsm_receiver's buffer
         * @return true if SCH burst is near, false otherwise
         */
        bool reach_sch_burst(const int nitems);

        /** Extracts channel impulse response from a SCH burst and computes first sample number of this burst
         *
         * @param input vector with input samples
         * @param chan_imp_resp complex vector where channel impulse response will be stored
         * @return number of first sample of the burst
         */
        int get_sch_chan_imp_resp(const gr_complex *input, gr_complex * chan_imp_resp);

        /** MLSE detection of a burst bits
         *
         * Detects bits of burst using viterbi algorithm.
         * @param input vector with input samples
         * @param chan_imp_resp vector with the channel impulse response
         * @param burst_start number of the first sample of the burst
         * @param output_binary vector with output bits
         */
        void detect_burst(const gr_complex * input, gr_complex * chan_imp_resp, int burst_start, unsigned char * output_binary);

        /** Encodes differentially input bits and maps them into MSK states
         *
         * @param input vector with input bits
         * @param nitems number of samples in the "input" vector
         * @param gmsk_output bits mapped into MSK states
         * @param start_point first state
         */
        void gmsk_mapper(const unsigned char * input, int nitems, gr_complex * gmsk_output, gr_complex start_point);

        /** Correlates MSK mapped sequence with input signal
         *
         * @param sequence MKS mapped sequence
         * @param length length of the sequence
         * @param input_signal vector with input samples
         * @return correlation value
         */
        gr_complex correlate_sequence(const gr_complex * sequence, int length, const gr_complex * input);

        /** Computes autocorrelation of input vector for positive arguments
         *
         * @param input vector with input samples
         * @param out output vector
         * @param nitems length of the input vector
         */
        inline void autocorrelation(const gr_complex * input, gr_complex * out, int nitems);

        /** Filters input signal through channel impulse response
         *
         * @param input vector with input samples
         * @param nitems number of samples to pass through filter
         * @param filter filter taps - channel impulse response
         * @param filter_length nember of filter taps
         * @param output vector with filtered samples
         */
        inline void mafi(const gr_complex * input, int nitems, gr_complex * filter, int filter_length, gr_complex * output);

        /**  Extracts channel impulse response from a normal burst and computes first sample number of this burst
         *
         * @param input vector with input samples
         * @param chan_imp_resp complex vector where channel impulse response will be stored
         * @param search_range possible absolute offset of a channel impulse response start
         * @param bcc base station color code - number of a training sequence
         * @return first sample number of normal burst
         */
        int get_norm_chan_imp_resp(const gr_complex *input, gr_complex * chan_imp_resp, float *corr_max, int bcc);

        /**
         * Sends burst through a C0 (for burst from C0 channel) or Cx (for other bursts) message port
         *
         * @param burst_nr - frame number of the burst
         * @param burst_binary - content of the burst
         * @b_type - type of the burst
         */
        void send_burst(burst_counter burst_nr, const unsigned char * burst_binary, uint8_t burst_type, unsigned int input_nr);

        /**
         * Configures burst types in different channels
         */
        void configure_receiver();







       void set_cell_allocation(const std::vector<int> &cell_allocation);
       void set_tseq_nums(const std::vector<int> & tseq_nums);
       void reset();




#endif /* MAIN_H_ */
