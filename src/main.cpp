/*
 * main.cpp
 *
 *  Created on: 17-Jul-2015
 *      Author: akib
 */

#include <convenience.h>
#include<rtl-sdr.h>
#include <stdlib.h>
#include<stdio.h>
#include<unistd.h>
#include<string.h>
#include <sys/types.h>
#include <signal.h>
#include <gnuradio/gr_complex.h>
#include<vector>
#include<liquid/liquid.h>
#include<rtl_cpp.h>
//#include <complex.h>
#include<uchar2float.h>
#include <gsm_constants.h>
#include<receiver.h>
#include<main.h>
#include <boost/circular_buffer.hpp>

#include <math.h>
#include <gnuradio/math.h>
#include <algorithm>
#include <numeric>
#include<viterbi_detector.h>

//#include <boost/scoped_ptr.hpp>

#include <string.h>
#include <iostream>
#include <iomanip>
std::vector<gr_complex> _lut;
#define DEFAULT_SAMPLE_RATE		2048000
#define DEFAULT_BUF_LENGTH		(16 * 16384)
#define MINIMAL_BUF_LENGTH		512
#define MAXIMAL_BUF_LENGTH		(256 * 16384)
gr_complex *outflt;
static int do_exit = 0;
static uint32_t bytes_to_read = 0;
static rtlsdr_dev_t *dev = NULL;

#define SYNC_SEARCH_RANGE 30
int PPM=0;

int samples_len=0;

states d_state=fcch_search;
using namespace std;
void usage(void)
{
	fprintf(stderr,
		"rtl_sdr, an I/Q recorder for RTL2832 based DVB-T receivers\n\n"
		"Usage:\t -f frequency_to_tune_to [Hz]\n"
		"\t[-s samplerate (default: 2048000 Hz)]\n"
		"\t[-d device_index (default: 0)]\n"
		"\t[-g gain (default: 0 for auto)]\n"
		"\t[-p ppm_error (default: 0)]\n"
		"\t[-b output_block_size (default: 16 * 16384)]\n"
		"\t[-n number of samples to read (default: 0, infinite)]\n"
		"\t[-S force sync output (default: async)]\n"
		"\tfilename (a '-' dumps samples to stdout)\n\n");
	exit(1);
}

#ifdef _WIN32
BOOL WINAPI
sighandler(int signum)
{
	if (CTRL_C_EVENT == signum) {
		fprintf(stderr, "Signal caught, exiting!\n");
		do_exit = 1;
		rtlsdr_cancel_async(dev);
		return TRUE;
	}
	return FALSE;
}
#else
static void sighandler(int signum)
{
	fprintf(stderr, "Signal caught, exiting!\n");
	do_exit = 1;

	rtlsdr_cancel_async(dev);
	exit(1);
}
#endif

void clock_offset_corrector(gr_complex* in,int len,double sample_rate,int osr){
/*
 * Check Global VAR PPM
 */
int const_ppm=PPM;

/*
 *

Const===|
		|_____Mul Const
				|_____Add Const
						|_____Fractional Resamp

		|_____Mul Const
				|_____Controlled Rotator
*/
/*
 * set samp rate out
 *
 */



/*
 * Set Parameters for Mul and add const based on samp rate
 */




}

void gmsk_mapper(const unsigned char * input, int nitems, gr_complex * gmsk_output, gr_complex start_point){

	  gr_complex j = gr_complex(0.0, 1.0);

	    int current_symbol;
	    int encoded_symbol;
	    int previous_symbol = 2 * input[0] - 1;
	    gmsk_output[0] = start_point;

	    for (int i = 1; i < nitems; i++)
	    {
	        //change bits representation to NRZ
	        current_symbol = 2 * input[i] - 1;
	        //differentially encode
	        encoded_symbol = current_symbol * previous_symbol;
	        //and do gmsk mapping
	        gmsk_output[i] = j * gr_complex(encoded_symbol, 0.0) * gmsk_output[i-1];
	        previous_symbol = current_symbol;
	    }
}
void configure_receiver()
{
    d_channel_conf.set_multiframe_type(TIMESLOT0, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT0, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);

    d_channel_conf.set_burst_types(TIMESLOT0, TEST_CCH_FRAMES, sizeof(TEST_CCH_FRAMES) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_burst_types(TIMESLOT0, FCCH_FRAMES, sizeof(FCCH_FRAMES) / sizeof(unsigned), fcch_burst);
    d_channel_conf.set_burst_types(TIMESLOT0, SCH_FRAMES, sizeof(SCH_FRAMES) / sizeof(unsigned), sch_burst);

    d_channel_conf.set_multiframe_type(TIMESLOT1, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT1, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_multiframe_type(TIMESLOT2, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT2, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_multiframe_type(TIMESLOT3, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT3, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_multiframe_type(TIMESLOT4, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT4, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_multiframe_type(TIMESLOT5, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT5, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_multiframe_type(TIMESLOT6, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT6, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
    d_channel_conf.set_multiframe_type(TIMESLOT7, multiframe_51);
    d_channel_conf.set_burst_types(TIMESLOT7, TEST51, sizeof(TEST51) / sizeof(unsigned), dummy_or_normal);
}
void init_gmsk(){
	 int i;
	                                                                      //don't send samples to the receiver until there are at least samples for one
	   // set_output_multiple(floor((TS_BITS + 2 * GUARD_PERIOD) * d_OSR)); // burst and two gurad periods (one gurard period is an arbitrary overlap)
	    gmsk_mapper(SYNC_BITS, N_SYNC_BITS, d_sch_training_seq, gr_complex(0.0, -1.0));
	    for (i = 0; i < TRAIN_SEQ_NUM; i++)
	    {
	        gr_complex startpoint = (train_seq[i][0]==0) ? gr_complex(1.0, 0.0) : gr_complex(-1.0, 0.0); //if first bit of the seqeunce ==0  first symbol ==1
	                                                                                                     //if first bit of the seqeunce ==1  first symbol ==-1
	        gmsk_mapper(train_seq[i], N_TRAIN_BITS, d_norm_training_seq[i], startpoint);
	    }

	    configure_receiver();  //configure the receiver - tell it where to find which burst type



}



void resampler(gr_complex *in,int len,double sample_rate,int osr)
{

	float samp_out=1625000.0/6.0*osr;



float r= samp_out/sample_rate;
float As=100.0f; // resampling filter stop-band attenuation [dB]
//unsigned int n= NUM_SAMPLES; // number of input samples

msresamp_crcf q = msresamp_crcf_create(r,As);
//msresamp_crcf_print(q);
float delay = msresamp_crcf_get_delay(q);

// number of input samples (zero-padded)
unsigned int nx = len + (int)ceilf(delay) + 10;

// output buffer with extra padding for good measure
unsigned int ny_alloc = (unsigned int) (2*(float)nx * r);  // allocation for output

//float complex r[nx]; // received signal
//float complex y[ny_alloc]; // resampled signal
gr_complex out[ny_alloc];
//while (1) {
    unsigned int ny;
    //read_complex_iq_block(r, NUM_SAMPLES); // made up function
    msresamp_crcf_execute(q, in, len, out, &ny);
    fwrite(out,sizeof(gr_complex),ny,file2);
    // here "y" is signal with sample rate of OUT_SAMPLERATE and "ny" shows how many samples are in "y"
//}
    int i=0;
    int first=0;
    //dump_complex(out,ny,1);


    if (d_state==synchronized){
 sync:   	 for (;i<=ny;){
    			  //d_counter=0;
    			    //dump_complex(out,70000,0);
    		    	gsm_decode(out,70000);

    		    	//consume_each(out,i)
    		    	//if(ctr>4)
    		    	//	exit(1);
    		    	i=i+625;
    		    	//ctr++;
    		    }
    }
    else{
    	 for (i=	70000;i<=ny;){
    			  //d_counter=0;
    			    //dump_complex(out,70000,0);
    		    	gsm_decode(out,70000);

    		    	//consume_each(out,i)
    		    	//if(ctr>4)
    		    	//	exit(1);
    		    	if (d_state==synchronized){
    		    		goto sync;
    		    	}
    		    	i=i+625;
    		    	//ctr++;

    		    }
    }



msresamp_crcf_destroy(q);

}

void  filter( gr_complex *in,int len){

	// options

	//fwrite(in,sizeof(gr_complex),len,file3);

	gr_complex out[DEFAULT_BUF_LENGTH];
	//out = (gr_complex *)malloc(DEFAULT_BUF_LENGTH * sizeof(gr_complex));
	double samplerate=1e6;
	double cutofffreq=125e3;

	float ft=0.1f; //filter transition
	float As = 120.0f; // stop-band attenuation
	float mu=0.0f;//fractional timing offset
	// estimate required filter length and generate filter
	unsigned int h_len = estimate_req_filter_len(ft,As);
	float h[h_len];
	//liquid_firdes_kaiser(h_len,fc,As,mu,h);


float fc = cutofffreq/samplerate; // cutoff frequency

//printf("\nLen=%d\n",h_len);



// design filter from prototype and scale to bandwidth
firfilt_crcf q = firfilt_crcf_create_kaiser(h_len, fc, As, 0.0f);
//firfilt_crcf_set_scale(q, 2.0f*cutofffreq);
int i=0;


for (i=0;i<len;i++){
	 firfilt_crcf_push(q, in[i]);
	 firfilt_crcf_execute(q, &out[i]);
}

resampler(out,len,1e6,4);
//fwrite(out,sizeof(gr_complex),len,file3);

    // here lowpass filtered signal "rf" can be given to demodulator as shown above
}
//firfilt_crcf_destroy(q);




bool find_fcch_burst(gr_complex *input, const int nitems, double & computed_freq_offset)
{
    boost::circular_buffer<float> phase_diff_buffer(FCCH_HITS_NEEDED * d_OSR); //circular buffer used to scan throug signal to find
    //best match for FCCH burst
    float phase_diff = 0;
    gr_complex conjprod;
    int start_pos = -1;
    int hit_count = 0;
    int miss_count = 0;
    float min_phase_diff;
    float max_phase_diff;
    double best_sum = 0;
    float lowest_max_min_diff = 99999;

    int to_consume = 0;
    int sample_number = 0;
    bool end = false;
    bool result = false;
    boost::circular_buffer<float>::iterator buffer_iter;

    /**@name Possible states of FCCH search algorithm*/
    //@{
    enum states
    {
        init,               ///< initialize variables
        search,             ///< search for positive samples
        found_something,    ///< search for FCCH and the best position of it
        fcch_found,         ///< when FCCH was found
        search_fail         ///< when there is no FCCH in the input vector
    } fcch_search_state;
    //@}

    fcch_search_state = init;

    while (!end)
    {
        switch (fcch_search_state)
        {

        case init: //initialize variables
            hit_count = 0;
            miss_count = 0;
            start_pos = -1;
            lowest_max_min_diff = 99999;
            phase_diff_buffer.clear();
            fcch_search_state = search;

            break;

        case search:                                                // search for positive samples
            sample_number++;

            if (sample_number > nitems - FCCH_HITS_NEEDED * d_OSR)   //if it isn't possible to find FCCH because
            {
                                                                       //there's too few samples left to look into,
                to_consume = sample_number;                            //don't do anything with those samples which are left
                                                                       //and consume only those which were checked
                fcch_search_state = search_fail;
            }
            else
            {
                phase_diff = compute_phase_diff(input[sample_number], input[sample_number-1]);

                if (phase_diff > 0)                                   //if a positive phase difference was found
                {
                    to_consume = sample_number;
                    fcch_search_state = found_something;                //switch to state in which searches for FCCH
                }
                else
                {
                    fcch_search_state = search;
                }
            }

            break;

        case found_something:  // search for FCCH and the best position of it
        {
            if (phase_diff > 0)
            {
                hit_count++;       //positive phase differencies increases hits_count
            }
            else
            {
                miss_count++;      //negative increases miss_count
            }

            if ((miss_count >= FCCH_MAX_MISSES * d_OSR) && (hit_count <= FCCH_HITS_NEEDED * d_OSR))
            {
                //if miss_count exceeds limit before hit_count
                fcch_search_state = init;       //go to init
                continue;
            }
            else if (((miss_count >= FCCH_MAX_MISSES * d_OSR) && (hit_count > FCCH_HITS_NEEDED * d_OSR)) || (hit_count > 2 * FCCH_HITS_NEEDED * d_OSR))
            {
                //if hit_count and miss_count exceeds limit then FCCH was found
                fcch_search_state = fcch_found;
                continue;
            }
            else if ((miss_count < FCCH_MAX_MISSES * d_OSR) && (hit_count > FCCH_HITS_NEEDED * d_OSR))
            {
                //find difference between minimal and maximal element in the buffer
                //for FCCH this value should be low
                //this part is searching for a region where this value is lowest
                min_phase_diff = * (min_element(phase_diff_buffer.begin(), phase_diff_buffer.end()));
                max_phase_diff = * (max_element(phase_diff_buffer.begin(), phase_diff_buffer.end()));

                if (lowest_max_min_diff > max_phase_diff - min_phase_diff)
                {
                    lowest_max_min_diff = max_phase_diff - min_phase_diff;
                    start_pos = sample_number - FCCH_HITS_NEEDED * d_OSR - FCCH_MAX_MISSES * d_OSR; //store start pos
                    best_sum = 0;

                    for (buffer_iter = phase_diff_buffer.begin();
                            buffer_iter != (phase_diff_buffer.end());
                            buffer_iter++)
                    {
                        best_sum += *buffer_iter - (M_PI / 2) / d_OSR;   //store best value of phase offset sum
                    }
                }
            }

            sample_number++;

            if (sample_number >= nitems)      //if there's no single sample left to check
            {
                fcch_search_state = search_fail;//FCCH search failed
                continue;
            }

            phase_diff = compute_phase_diff(input[sample_number], input[sample_number-1]);
            phase_diff_buffer.push_back(phase_diff);
            fcch_search_state = found_something;
        }
        break;

        case fcch_found:
        {
            to_consume = start_pos + FCCH_HITS_NEEDED * d_OSR + 1; //consume one FCCH burst

            d_fcch_start_pos = d_counter + start_pos;

            //compute frequency offset
            double phase_offset = best_sum / FCCH_HITS_NEEDED;
            double freq_offset = phase_offset * 1625000.0/6 / (2 * M_PI); //1625000.0/6 - GMSK symbol rate in GSM
            computed_freq_offset = freq_offset;

            end = true;
            result = true;
            break;
        }

        case search_fail:
            end = true;
            result = false;
            break;
        }
    }

    d_counter += to_consume;
    //consume_each(to_consume);
consume_each(input,to_consume);
samples_len=nitems-to_consume;
    //fprintf(stdout,"\nFCCH To Consume %d",to_consume);
    return result;
}

double compute_freq_offset(const gr_complex * input, unsigned first_sample, unsigned last_sample)
{
    double phase_sum = 0;
    unsigned ii;

    for (ii = first_sample; ii < last_sample; ii++)
    {
        double phase_diff = compute_phase_diff(input[ii], input[ii-1]) - (M_PI / 2) / d_OSR;
        phase_sum += phase_diff;
    }

    double phase_offset = phase_sum / (last_sample - first_sample);
    double freq_offset = phase_offset * 1625000.0 / (12.0 * M_PI);
    return freq_offset;
}

inline float compute_phase_diff(gr_complex val1, gr_complex val2)
{
    gr_complex conjprod = val1 * conj(val2);
   // return fast_atan2f(imag(conjprod), real(conjprod));
    return gr::fast_atan2f(imag(conjprod), real(conjprod));

}
void gsm_decode(gr_complex* input,int noutput_items)
{
	init_gmsk();
	int z=0;
//    std::vector<const gr_complex *> iii = (std::vector<const gr_complex *>) input_items; // jak zrobić to rzutowanie poprawnie
    //gr_complex * input = (gr_complex *) input_items[0];
    //std::vector<tag_t> freq_offset_tags;
    //uint64_t start = nitems_read(0);
    //uint64_t stop = start + noutput_items;

   // float current_time = static_cast<float>(start)/(GSM_SYMBOL_RATE*d_OSR);
   // if((current_time - d_last_time) > 0.1)
   // {
   //     pmt::pmt_t msg = pmt::make_tuple(pmt::mp("current_time"),pmt::from_double(current_time));
   //     message_port_pub(pmt::mp("measurements"), msg);
   //     d_last_time = current_time;
   // }

    //pmt::pmt_t key = pmt::string_to_symbol("setting_freq_offset");
    //get_tags_in_range(freq_offset_tags, 0, start, stop, key);
    //bool freq_offset_tag_in_fcch = false;
    //uint64_t tag_offset=-1; //-1 - just some clearly invalid value

    /*if(!freq_offset_tags.empty()){
        tag_t freq_offset_tag = freq_offset_tags[0];
        tag_offset = freq_offset_tag.offset - start;

        burst_type b_type = d_channel_conf.get_burst_type(d_burst_nr);
        if(d_state == synchronized && b_type == fcch_burst){
            uint64_t last_sample_nr = ceil((GUARD_PERIOD + 2.0 * TAIL_BITS + 156.25) * d_OSR) + 1;
            if(tag_offset < last_sample_nr){
                freq_offset_tag_in_fcch = true;
            }
            d_freq_offset_setting = pmt::to_double(freq_offset_tag.value);
        } else {
            d_freq_offset_setting = pmt::to_double(freq_offset_tag.value);
        }
    }*/
//for (z=0;z<noutput_items;z++){
	switch (d_state)
	    {
	        //bootstrapping
	    case fcch_search:
	    {
	        double freq_offset_tmp;
	        if (find_fcch_burst(input, noutput_items,freq_offset_tmp))
	        {
	            //pmt::pmt_t msg = pmt::make_tuple(pmt::mp("freq_offset"),pmt::from_double(freq_offset_tmp-d_freq_offset_setting),pmt::mp("fcch_search"));
	           // message_port_pub(pmt::mp("measurements"), msg);
	        	fprintf(stderr,"found fcch offset is %f\n",freq_offset_tmp);
	            d_state = sch_search;
	        }
	        else
	        {
	            d_state = fcch_search;
	        }
	        break;
	    }

	    case sch_search:
	    {
	    	//fprintf(stderr,"found sch\n");
	        std::vector<gr_complex> channel_imp_resp(CHAN_IMP_RESP_LENGTH*d_OSR);
	        int t1, t2, t3;
	        int burst_start = 0;
	        unsigned char output_binary[BURST_SIZE];

	        if (reach_sch_burst(input,noutput_items))                                //wait for a SCH burst
	        {
	        	//fprintf(stderr,"inside sch search\n");
	            burst_start = get_sch_chan_imp_resp(input, &channel_imp_resp[0]); //get channel impulse response from it
	            detect_burst(input, &channel_imp_resp[0], burst_start, output_binary); //detect bits using MLSE detection
	            if (decode_sch(&output_binary[3], &t1, &t2, &t3, &d_ncc, &d_bcc) == 0)   //decode SCH burst
	            {
	                d_burst_nr.set(t1, t2, t3, 0);                                  //set counter of bursts value
	                d_burst_nr++;

	                consume_each(input,burst_start + BURST_SIZE * d_OSR + 4*d_OSR);   //consume samples up to next guard period
	                d_state = synchronized;
	                fprintf(stderr,"synchronized\n");
	            }
	            else
	            {
	                d_state = fcch_search;                       //if there is error in the sch burst go back to fcch search phase
	            }
	        }
	        else
	        {
	            d_state = sch_search;
	        }
	        break;
	    }
	    //in this state receiver is synchronized and it processes bursts according to burst type for given burst number
	    case synchronized:
	    {
	    	///fprintf(dump_processed,"=========================");
	    	//dump_complex(input,70000,0);
	    	//ctr++;
	    	//if (ctr>2)
	    	//	exit(1);
	        std::vector<gr_complex> channel_imp_resp(CHAN_IMP_RESP_LENGTH*d_OSR);
	        int offset = 0;
	        int to_consume = 0;
	        unsigned char output_binary[BURST_SIZE];

	        burst_type b_type;

	        for(int input_nr=0; input_nr<1; input_nr++)
	        {
	            double signal_pwr = 0;
	            //input = (gr_complex *)input_items[input_nr];

	            for(int ii=GUARD_PERIOD;ii<TS_BITS;ii++)
	            {
	                signal_pwr += abs(input[ii])*abs(input[ii]);
	            }
	            signal_pwr = signal_pwr/(TS_BITS);
	            d_signal_dbm = round(10*log10(signal_pwr/50));
	            if(input_nr==0){
	                d_c0_signal_dbm = d_signal_dbm;
	            }

	            if(input_nr==0) //for c0 channel burst type is controlled by channel configuration
	            {
	                b_type = d_channel_conf.get_burst_type(d_burst_nr); //get burst type for given burst number
	            }
	            else
	            {
	                b_type = normal_or_noise; //for the rest it can be only normal burst or noise (at least at this moment of development)
	            }

	            switch (b_type)
	            {
	            case fcch_burst:                                                                      //if it's FCCH  burst
	            {
	            	fprintf(stderr,"\nfcch_burst!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	            	d_counter_bursts++;
	                const unsigned first_sample = ceil((GUARD_PERIOD + 2 * TAIL_BITS) * d_OSR) + 1;
	                const unsigned last_sample = first_sample + USEFUL_BITS * d_OSR - TAIL_BITS * d_OSR;
	                double freq_offset_tmp = compute_freq_offset(input, first_sample, last_sample);       //extract frequency offset from it

	                //send_burst(d_burst_nr, fc_fb, GSMTAP_BURST_FCCH, input_nr);

	                //pmt::pmt_t msg = pmt::make_tuple(pmt::mp("freq_offset"),pmt::from_double(freq_offset_tmp-d_freq_offset_setting),pmt::mp("synchronized"));
	                //message_port_pub(pmt::mp("measurements"), msg);
	                break;
	            }
	            case sch_burst:                                                                      //if it's SCH burst
	            {	d_counter_bursts++;
	            fprintf(stderr,"\nsch_burst!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	                int t1, t2, t3, d_ncc, d_bcc;
	                d_c0_burst_start = get_sch_chan_imp_resp(input, &channel_imp_resp[0]);                //get channel impulse response

	                detect_burst(input, &channel_imp_resp[0], d_c0_burst_start, output_binary);           //MLSE detection of bits
	                //send_burst(d_burst_nr, output_binary, GSMTAP_BURST_SCH, input_nr);
	                if (decode_sch(&output_binary[3], &t1, &t2, &t3, &d_ncc, &d_bcc) == 0)           //and decode SCH data
	                {
	                    // d_burst_nr.set(t1, t2, t3, 0);                                              //but only to check if burst_start value is correct
	                    d_failed_sch = 0;
	                    offset =  d_c0_burst_start - floor((GUARD_PERIOD) * d_OSR);                         //compute offset from burst_start - burst should start after a guard period
	                    to_consume += offset;                                                          //adjust with offset number of samples to be consumed
	                }
	                else
	                {
	                    d_failed_sch++;
	                    if (d_failed_sch >= MAX_SCH_ERRORS)
	                    {
	                        d_state = fcch_search;
	                        //pmt::pmt_t msg = pmt::make_tuple(pmt::mp("freq_offset"),pmt::from_double(0.0),pmt::mp("sync_loss"));
	                        //message_port_pub(pmt::mp("measurements"), msg);
	                        fprintf(stderr,"Re-Synchronization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	                    }
	                }
	                break;
	            }
	            case normal_burst:
	            {
	            	d_counter_bursts++;
	            	fprintf(stderr,"\nNormal burst!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	                float normal_corr_max;                                                    //if it's normal burst
	                d_c0_burst_start = get_norm_chan_imp_resp(input, &channel_imp_resp[0], &normal_corr_max, d_bcc); //get channel impulse response for given training sequence number - d_bcc
	                detect_burst(input, &channel_imp_resp[0], d_c0_burst_start, output_binary);            //MLSE detection of bits
	                //send_burst(d_burst_nr, output_binary, GSMTAP_BURST_NORMAL, input_nr);
	                break;
	            }
	            case dummy_or_normal:
	            {
	            	d_counter_bursts++;
	            	fprintf(stderr,"\nDummy or Normal burst!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	                unsigned int normal_burst_start, dummy_burst_start;
	                float dummy_corr_max, normal_corr_max;

	                dummy_burst_start = get_norm_chan_imp_resp(input, &channel_imp_resp[0], &dummy_corr_max, TS_DUMMY);
	                normal_burst_start = get_norm_chan_imp_resp(input, &channel_imp_resp[0], &normal_corr_max, d_bcc);

	                if (normal_corr_max > dummy_corr_max)
	                {
	                    d_c0_burst_start = normal_burst_start;
	                    detect_burst(input, &channel_imp_resp[0], normal_burst_start, output_binary);
	                    send_burst(d_burst_nr, output_binary);
	                }
	                else
	                {
	                    d_c0_burst_start = dummy_burst_start;
	                    send_burst(d_burst_nr, dummy_burst);
	                }
	                break;
	            }
	            case rach_burst:
	                break;
	            case dummy:
	            	d_counter_bursts++;
	                //send_burst(d_burst_nr, dummy_burst, GSMTAP_BURST_DUMMY, input_nr);
	                break;
	            case normal_or_noise:
	            {
	            	d_counter_bursts++;
	            	fprintf(stderr,"\n Noise or Normal burst!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	                unsigned int burst_start;
	                float normal_corr_max_tmp;
	                float normal_corr_max=-1e6;
	                int max_tn;
	                //std::vector<gr_complex> v(input, input + noutput_items);
	                if(d_signal_dbm>=d_c0_signal_dbm-13)
	                {
	                    if(d_tseq_nums.size()==0)              //there is no information about training sequence
	                    {                                      //however the receiver can detect it
	                        get_norm_chan_imp_resp(input, &channel_imp_resp[0], &normal_corr_max, 0);
	                        float ts_max=normal_corr_max;     //with use of a very simple algorithm based on finding
	                        int ts_max_num=0;                 //maximum correlation
	                        for(int ss=1; ss<=7; ss++)
	                        {
	                            get_norm_chan_imp_resp(input, &channel_imp_resp[0], &normal_corr_max, ss);
	                            if(ts_max<normal_corr_max)
	                            {
	                                ts_max = normal_corr_max;
	                                ts_max_num = ss;
	                            }
	                        }
	                        d_tseq_nums.push_back(ts_max_num);
	                    }
	                    int tseq_num;
	                    if(input_nr<=d_tseq_nums.size()){
	                        tseq_num = d_tseq_nums[input_nr-1];
	                    } else {
	                        tseq_num = d_tseq_nums.back();
	                    }
	                    burst_start = get_norm_chan_imp_resp(input, &channel_imp_resp[0], &normal_corr_max, tseq_num);
	//                  if(abs(d_c0_burst_start-burst_start)<=2){ //unused check/filter based on timing
	                    if((normal_corr_max/sqrt(signal_pwr))>=0.9){
	                        detect_burst(input, &channel_imp_resp[0], burst_start, output_binary);
	                        //send_burst(d_burst_nr, output_binary, GSMTAP_BURST_NORMAL, input_nr);
	                    }
	                }
	                break;
	            }
	            case empty:   //if it's empty burst
	                break;      //do nothing
	            }

	          //  if(input_nr==input_items.size()-1)
	          // {
	                d_burst_nr++;   //go to next burst
	                to_consume += TS_BITS * d_OSR + d_burst_nr.get_offset();  //consume samples of the burst up to next guard period
	                fprintf(stderr,"\nto consume in sync-%d---%d",to_consume,d_burst_nr.get_offset());
	                consume_each(input,to_consume);
	          //  }
	            //and add offset which is introduced by
	            //0.25 fractional part of a guard period
	        }
	    }
	    break;
	    }
//}

    //return 0;
}

void send_burst(burst_counter burst_nr, const unsigned char * burst_binary){
	fprintf(stdout,"\nframe-nr= %x",d_burst_nr.get_frame_nr());
	for(int i=0;i<BURST_SIZE;i++){
		fprintf(stdout,"%x",burst_binary[i]);
	}
}
//computes autocorrelation for positive arguments
inline void autocorrelation(const gr_complex * input, gr_complex * out, int nitems)
{
    int i, k;
    for (k = nitems - 1; k >= 0; k--)
    {
        out[k] = gr_complex(0, 0);
        for (i = k; i < nitems; i++)
        {
            out[k] += input[i] * conj(input[i-k]);
        }
    }
}


inline void mafi(const gr_complex * input, int nitems, gr_complex * filter, int filter_length, gr_complex * output)
{
    int ii = 0, n, a;

    for (n = 0; n < nitems; n++)
    {
        a = n * d_OSR;
        output[n] = 0;
        ii = 0;

        while (ii < filter_length)
        {
            if ((a + ii) >= nitems*d_OSR){
                break;
            }
            output[n] += input[a+ii] * filter[ii];
            ii++;
        }
    }
}

void detect_burst(const gr_complex * input, gr_complex * chan_imp_resp, int burst_start, unsigned char * output_binary)
{
    float output[BURST_SIZE];
    std::vector<gr_complex> rhh_temp(CHAN_IMP_RESP_LENGTH*d_OSR);
    gr_complex rhh[CHAN_IMP_RESP_LENGTH];
    gr_complex filtered_burst[BURST_SIZE];
    int start_state = 3;
    unsigned int stop_states[2] = {4, 12};

    autocorrelation(chan_imp_resp, &rhh_temp[0], d_chan_imp_length*d_OSR);
    for (int ii = 0; ii < (d_chan_imp_length); ii++)
    {
        rhh[ii] = conj(rhh_temp[ii*d_OSR]);
    }

    mafi(&input[burst_start], BURST_SIZE, chan_imp_resp, d_chan_imp_length*d_OSR, filtered_burst);

    viterbi_detector(filtered_burst, BURST_SIZE, rhh, start_state, stop_states, 2, output);

    //fprintf(stdout,"\n");
    for (int i = 0; i < BURST_SIZE ; i++)
    {
    	//fprintf(stdout,"%x",(output[i] > 0));
        output_binary[i] = (output[i] > 0);
    }
}

gr_complex correlate_sequence(const gr_complex * sequence, int length, const gr_complex * input)
{
    gr_complex result(0.0, 0.0);
    int sample_number = 0;

    for (int ii = 0; ii < length; ii++)
    {
        sample_number = (ii * d_OSR) ;
        result += sequence[ii] * conj(input[sample_number]);
    }

    result = result / gr_complex(length, 0);
    return result;
}
int get_sch_chan_imp_resp(const gr_complex *input, gr_complex * chan_imp_resp)
{
    std::vector<gr_complex> correlation_buffer;
    std::vector<float> power_buffer;
    std::vector<float> window_energy_buffer;

    int strongest_window_nr;
    int burst_start = 0;
    int chan_imp_resp_center = 0;
    float max_correlation = 0;
    float energy = 0;

    for (int ii = SYNC_POS * d_OSR; ii < (SYNC_POS + SYNC_SEARCH_RANGE) *d_OSR; ii++)
    {
        gr_complex correlation = correlate_sequence(&d_sch_training_seq[5], N_SYNC_BITS - 10, &input[ii]);
        correlation_buffer.push_back(correlation);
        power_buffer.push_back(std::pow(abs(correlation), 2));
    }
    //compute window energies
    std::vector<float>::iterator iter = power_buffer.begin();
    bool loop_end = false;
    while (iter != power_buffer.end())
    {
        std::vector<float>::iterator iter_ii = iter;
        energy = 0;

        for (int ii = 0; ii < (d_chan_imp_length) *d_OSR; ii++, iter_ii++)
        {
            if (iter_ii == power_buffer.end())
            {
                loop_end = true;
                break;
            }
            energy += (*iter_ii);
        }
        if (loop_end)
        {
            break;
        }
        iter++;
        window_energy_buffer.push_back(energy);
    }

    strongest_window_nr = max_element(window_energy_buffer.begin(), window_energy_buffer.end()) - window_energy_buffer.begin();
    //   d_channel_imp_resp.clear();

    max_correlation = 0;
    for (int ii = 0; ii < (d_chan_imp_length) *d_OSR; ii++)
    {
        gr_complex correlation = correlation_buffer[strongest_window_nr + ii];
        if (abs(correlation) > max_correlation)
        {
            chan_imp_resp_center = ii;
            max_correlation = abs(correlation);
        }
        //     d_channel_imp_resp.push_back(correlation);
        chan_imp_resp[ii] = correlation;
    }

    burst_start = strongest_window_nr + chan_imp_resp_center - 48 * d_OSR - 2 * d_OSR + 2 + SYNC_POS * d_OSR;
    return burst_start;
}

bool reach_sch_burst(gr_complex *in,const int nitems)
{
    //it just consumes samples to get near to a SCH burst
    int to_consume = 0;
    bool result = false;
    unsigned sample_nr_near_sch_start = d_fcch_start_pos + (FRAME_BITS - SAFETY_MARGIN) * d_OSR;

    //consume samples until d_counter will be equal to sample_nr_near_sch_start
    if (d_counter < sample_nr_near_sch_start)
    {
        if (d_counter + nitems >= sample_nr_near_sch_start)
        {
            to_consume = sample_nr_near_sch_start - d_counter;
        }
        else
        {
            to_consume = nitems;
        }
        result = false;
    }
    else
    {
        to_consume = 0;
        result = true;
    }
fprintf(stdout,"\nTo Consume %d",to_consume);
    d_counter += to_consume;
    samples_len=nitems-to_consume;
    consume_each(in,to_consume);
    return result;
}
int consume_each(gr_complex *in,int to_consume){
int i;
int counter=to_consume;
for (i=0;i<DEFAULT_BUF_LENGTH/2;i++){
	in[i]=in[counter];
	counter++;
}
return to_consume;

}


//especially computations of strongest_window_nr
int get_norm_chan_imp_resp(const gr_complex *input, gr_complex * chan_imp_resp, float *corr_max, int bcc)
{
    std::vector<gr_complex> correlation_buffer;
    std::vector<float> power_buffer;
    std::vector<float> window_energy_buffer;

    int strongest_window_nr;
    int burst_start = 0;
    int chan_imp_resp_center = 0;
    float max_correlation = 0;
    float energy = 0;

    int search_center = (int)((TRAIN_POS + GUARD_PERIOD) * d_OSR);
    int search_start_pos = search_center + 1 - 5*d_OSR;
    //   int search_start_pos = search_center -  d_chan_imp_length * d_OSR;
    int search_stop_pos = search_center + d_chan_imp_length * d_OSR + 5 * d_OSR;

    for(int ii = search_start_pos; ii < search_stop_pos; ii++)
    {
        gr_complex correlation = correlate_sequence(&d_norm_training_seq[bcc][TRAIN_BEGINNING], N_TRAIN_BITS - 10, &input[ii]);
        correlation_buffer.push_back(correlation);
        power_buffer.push_back(std::pow(abs(correlation), 2));
    }
//    plot(power_buffer);
    //compute window energies
    std::vector<float>::iterator iter = power_buffer.begin();
    bool loop_end = false;
    while (iter != power_buffer.end())
    {
        std::vector<float>::iterator iter_ii = iter;
        energy = 0;

        for (int ii = 0; ii < (d_chan_imp_length - 2)*d_OSR; ii++, iter_ii++)
        {
            if (iter_ii == power_buffer.end())
            {
                loop_end = true;
                break;
            }
            energy += (*iter_ii);
        }
        if (loop_end)
        {
            break;
        }
        iter++;

        window_energy_buffer.push_back(energy);
    }

    strongest_window_nr = max_element(window_energy_buffer.begin(), window_energy_buffer.end()-((d_chan_imp_length)*d_OSR)) - window_energy_buffer.begin();
    //strongest_window_nr = strongest_window_nr-d_OSR;
    if(strongest_window_nr<0){
       strongest_window_nr = 0;
    }

    max_correlation = 0;
    for (int ii = 0; ii < (d_chan_imp_length)*d_OSR; ii++)
    {
        gr_complex correlation = correlation_buffer[strongest_window_nr + ii];
        if (abs(correlation) > max_correlation)
        {
            chan_imp_resp_center = ii;
            max_correlation = abs(correlation);
        }
        //     d_channel_imp_resp.push_back(correlation);
        chan_imp_resp[ii] = correlation;
    }

    *corr_max = max_correlation;

    //DCOUT("strongest_window_nr_new: " << strongest_window_nr);
    burst_start = search_start_pos + strongest_window_nr - TRAIN_POS * d_OSR; //compute first sample posiiton which corresponds to the first sample of the impulse response

    //DCOUT("burst_start: " << burst_start);
    return burst_start;
}


void pushasync(unsigned char *buf,uint32_t len){
	int y;
	int16_t buffer[DEFAULT_BUF_LENGTH];
	gr_complex out[DEFAULT_BUF_LENGTH];
	unsigned short  _buf1[DEFAULT_BUF_LENGTH];
	for(y=0;y<len;y++)
	{
		buffer[y]=(int16_t)buf[y] - 127;
	}



			memcpy(_buf1, buf, DEFAULT_BUF_LENGTH);
			for (y=0; y < len; ++y)
				out[y] = _lut[ *(_buf1 + y) ];

			fwrite(out,sizeof(gr_complex),len/2,file);
}

unsigned char * deinterleave(unsigned char* in,uint32_t len,int even){
	int i,y,z;
	unsigned char buf[DEFAULT_BUF_LENGTH/2];
	y=0;z=0;

	for (i=0;i<len;i++){
		if(even==1){
			if((i%2)==0){
				buf[y]=in[i];
				y++;
			}

		}
		else{
			if((i%2)!=0){
				buf[y]=in[i];
				y++;
			}


		}


	}
	return buf;
}
void uchar2float_wrapper(unsigned char *img,unsigned char *real,float *imgflt,float * realflt,uint32_t len){

}


float * add_constant(float* buf,uint32_t len){
	float out[len];
	int i;
	for(i=0;i<len;i++){
		out[i]= buf[i]-127;
	}
	return out;
}

float * mul_constant(float* buf,uint32_t len){
	int i;
	for(i=0;i<len;i++){
		buf[i]=buf[i]*0.008;
	}
	return buf;
}

gr_complex * float2complex(float*img,float* real,uint32_t len){
	size_t j;
	gr_complex out[len];
	for (size_t j = 0; j < len; j++)
	   out[j] = gr_complex (real[j], img[j]);

	return out;
}
static void rtlsdr_callback(unsigned char *buf, uint32_t len, void *ctx)
{
	int y;
	if (ctx) {
		if (do_exit)
			return;

		if ((bytes_to_read > 0) && (bytes_to_read < len)) {
			len = bytes_to_read;
			do_exit = 1;
			rtlsdr_cancel_async(dev);
		}
		pushasync(buf,len);
		//filter( out,n_read/2);
		//if (fwrite(buf, 1, len, (FILE*)ctx) != len) {
		//	fprintf(stderr, "Short write, samples lost, exiting!\n");
		//	rtlsdr_cancel_async(dev);
		///}

		if (bytes_to_read > 0)
			bytes_to_read -= len;
	}
}
void init_lut(){
	int i;
	  for ( i = 0; i <= 0xffff; i++) {
	#ifdef BOOST_LITTLE_ENDIAN
	    _lut.push_back( gr_complex( (float(i & 0xff) - 127.4f) * (1.0f/128.0f),
	                                (float(i >> 8) - 127.4f) * (1.0f/128.0f) ) );
	#else // BOOST_BIG_ENDIAN
	    _lut.push_back( gr_complex( (float(i >> 8) - 127.4f) * (1.0f/128.0f),
	                                (float(i & 0xff) - 127.4f) * (1.0f/128.0f) ) );
	#endif
	  }

}
void pass_samples(unsigned char * buf,uint32_t len)
{
	unsigned char *img,*real,*temp;
	uint32_t complex_len=len/2;

	img=(unsigned char *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(unsigned char));
	real=(unsigned char *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(unsigned char));
	temp=deinterleave(buf,len,0);
	memcpy(img,temp,sizeof(unsigned char)* len/2);
	temp=deinterleave(buf,len,1);
	memcpy(real,temp,sizeof(unsigned char)* len/2);


	float *imgflt,*realflt,*tempfloat;

	imgflt = (float *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(float));
	realflt = (float *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(float));
	tempfloat = (float *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(float));


	tempfloat=uchar_array_to_float(img,complex_len);
	memcpy(imgflt,tempfloat,sizeof(float)*len/2);

	tempfloat=uchar_array_to_float(real,complex_len);
	memcpy(realflt,tempfloat,sizeof(float)*len/2);



	//printf("break");
	tempfloat=add_constant(imgflt,complex_len);
	memcpy(imgflt,tempfloat,sizeof(float)*len/2);
	tempfloat = (float *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(float));

	tempfloat=add_constant(realflt,complex_len);
	memcpy(realflt,tempfloat,sizeof(float)*len/2);


	tempfloat=mul_constant(imgflt,complex_len);
	memcpy(imgflt,tempfloat,sizeof(float)*len/2);
	tempfloat = (float *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(float));

	tempfloat=mul_constant(realflt,complex_len);
	memcpy(realflt,tempfloat,sizeof(float)*len/2);

	//mul_constant(imgflt,complex_len);
	//mul_constant(realflt,complex_len);

	 gr_complex *out;
	 //char * tempchar = (char *)malloc(DEFAULT_BUF_LENGTH/2 * sizeof(char));

	 out=float2complex(imgflt,realflt,complex_len);
	 outflt = (gr_complex *)malloc(complex_len* sizeof(gr_complex));
	 memcpy(outflt,out,sizeof(gr_complex)*complex_len);
	//fwrite(out,sizeof(gr_complex),complex_len,file3);
	 filter(outflt,complex_len);


}

void dump_complex(gr_complex * in,int len,int flag)
{
	int i;float re,im;
	if (flag==1){

		for (i=0;i<len;i++){
			re=in[i].real();
			im=in[i].imag();
			fprintf(dump_direct,"\n");
			fprintf(dump_direct,"%f---%f",re,im);
		}
	}
	else{

		for (i=0;i<len;i++){
			re=in[i].real();
			im=in[i].imag();
			fprintf(dump_processed,"\n");
			fprintf(dump_processed,"%f---%f",re,im);
		}
	}


}
int main(int argc, char **argv)
{


typedef std::complex<float>			gr_complex;
typedef std::complex<double>			gr_complexd;
#ifndef _WIN32
	struct sigaction sigact;
#endif
	char *filename = NULL;
	dump_direct=fopen("dump_direct","w");
	dump_processed=fopen("dump_processed","w");
	int n_read;
	int y = 0;
	int r, opt;
	int gain = 0;
	int ppm_error = 0;
	int sync_mode = 0;
	int read_iq=0;
	unsigned short *_buf;

	uint8_t *buffer;
	int dev_index = 0;
	int dev_given = 0;
	uint32_t frequency = 100000000;
	uint32_t samp_rate = DEFAULT_SAMPLE_RATE;
	uint32_t out_block_size = DEFAULT_BUF_LENGTH;
	d_state=fcch_search;


	while ((opt = getopt(argc, argv, "d:f:g:s:b:n:p:S:R")) != -1) {
		switch (opt) {
		case 'd':
			dev_index = verbose_device_search(optarg);
			dev_given = 1;
			break;
		case 'f':
			frequency = (uint32_t)atofs(optarg);
			break;
		case 'g':
			gain = (int)(atof(optarg) * 10); /* tenths of a dB */
			break;
		case 's':
			samp_rate = (uint32_t)atofs(optarg);
			break;
		case 'p':
			ppm_error = atoi(optarg);
			break;
		case 'b':
			out_block_size = (uint32_t)atof(optarg);
			break;
		case 'n':
			bytes_to_read = (uint32_t)atof(optarg) * 2;
			break;
		case 'S':
			sync_mode = 1;
			break;
		case 'R':
			read_iq=1;
			break;
		default:
			usage();
			break;
		}
	}

	if (argc <= optind) {
		usage();
	} else {
		filename = argv[optind];
		fprintf(stderr,"File Name %s",filename);
	}

	if(out_block_size < MINIMAL_BUF_LENGTH ||
	   out_block_size > MAXIMAL_BUF_LENGTH ){
		fprintf(stderr,
			"Output block size wrong value, falling back to default\n");
		fprintf(stderr,
			"Minimal length: %u\n", MINIMAL_BUF_LENGTH);
		fprintf(stderr,
			"Maximal length: %u\n", MAXIMAL_BUF_LENGTH);
		out_block_size = DEFAULT_BUF_LENGTH;
	}

	buffer = (uint8_t*)malloc(out_block_size * sizeof(uint8_t));




_buf=(unsigned short *)malloc(out_block_size * sizeof(unsigned short));
unsigned int i;

//printf("readiq=%d",read_iq);


init_lut();

if(!read_iq)
	{
     	if (!dev_given) {
			dev_index = verbose_device_search("0");
		}

		if (dev_index < 0) {
			exit(1);
		}

		r = rtlsdr_open(&dev, (uint32_t)dev_index);
		if (r < 0) {
			fprintf(stderr, "Failed to open rtlsdr device #%d.\n", dev_index);
			exit(1);
		}
	}

#ifndef _WIN32
	sigact.sa_handler = sighandler;
	sigemptyset(&sigact.sa_mask);
	sigact.sa_flags = 0;
	sigaction(SIGINT, &sigact, NULL);
	sigaction(SIGTERM, &sigact, NULL);
	sigaction(SIGQUIT, &sigact, NULL);
	sigaction(SIGPIPE, &sigact, NULL);
#else
	SetConsoleCtrlHandler( (PHANDLER_ROUTINE) sighandler, TRUE );
#endif

	if(!read_iq){
		/* Set the sample rate */
		verbose_set_sample_rate(dev, samp_rate);

		/* Set the frequency */
		verbose_set_frequency(dev, frequency);

		if (0 == gain) {
			 /* Enable automatic gain */
			verbose_auto_gain(dev);
		} else {
			/* Enable manual gain */
			gain = nearest_gain(dev, gain);
			verbose_gain_set(dev, gain);
		}

		verbose_ppm_set(dev, ppm_error);

		/* Reset endpoint before we start reading from it (mandatory) */
		verbose_reset_buffer(dev);

	}

	if(strcmp(filename, "-") == 0) { /* Write samples to stdout */
		file = stdout;
#ifdef _WIN32
		_setmode(_fileno(stdin), _O_BINARY);
#endif
	} else {
		if(read_iq){
			file = fopen(filename, "rb");

		}
		else
		{
			file = fopen(filename, "wb");

		}

		file2= fopen("test2.cfile","wb");
		file3= fopen("test3.cfile","wb");
		if (!file) {
			fprintf(stderr, "Failed to open %s\n", filename);
			goto out1;
		}
	}



float re,im;

	if (sync_mode) {

		fprintf(stderr, "Reading samples in sync mode...\n");
		while (!do_exit) {
			r = rtlsdr_read_sync(dev, buffer, out_block_size, &n_read);
			if (r < 0) {
				fprintf(stderr, "WARNING: sync read failed.\n");
				break;
			}
			gr_complex out[n_read];
			//out = (gr_complex *)malloc(out_block_size * sizeof(gr_complex));
			if ((bytes_to_read > 0) && (bytes_to_read < (uint32_t)n_read)) {
				n_read = bytes_to_read;
				do_exit = 1;
			}

			//pass_samples(buffer,n_read);

			memcpy(_buf, buffer, n_read);

			for (y=0; y < n_read; ++y)
			        out[y] = _lut[ *(_buf + y) ];

			fwrite(out,sizeof(gr_complex),n_read/2,file);
			filter( out,n_read/2);



			//if (fwrite(buffer, 1, n_read, file) != (size_t)n_read) {
			//	fprintf(stderr, "Short write, samples lost, exiting!\n");
			//	break;
			//}

			if ((uint32_t)n_read < out_block_size) {
				fprintf(stderr, "Short read, samples lost, exiting!\n");
				break;
			}

			if (bytes_to_read > 0)
				bytes_to_read -= n_read;
		}
	}else if (read_iq==1){
		//int n_read;


		do{
			//fprintf(stderr, "before read\n");
			n_read=fread(buffer,sizeof(uint8_t),out_block_size*sizeof(uint8_t),file);
			//fprintf(stderr, "after read=%d\n",n_read);
			if(n_read!=0){
				gr_complex out[n_read];
			//	fprintf(stderr, "passing\n",n_read);
				memcpy(_buf, buffer, n_read);

				for (y=0; y < n_read; ++y)
				{
					out[y] = _lut[ *(_buf + y) ];
					re=out[y].imag();
					im=out[y].real();
					out[y]=gr_complex(re,im);

				}

				//fwrite(out,sizeof(gr_complex),n_read/2,file);
				//consume_each(out,12);
				fprintf(stderr,"\nSending Samples");
				filter( out,n_read/2);


				//pass_samples(buffer,n_read);
			}


		}while ( n_read >= 1) ;// expecting 1

	}



	else {

		fprintf(stderr, "Reading samples in async mode...\n");
		r = rtlsdr_read_async(dev, rtlsdr_callback, (void *)file,
				      0, out_block_size);
	}

	if (do_exit)
		fprintf(stderr, "\nUser cancel, exiting...\n");
	else if (read_iq)
		fprintf(stderr, "\nExiting...\n");
	else
		fprintf(stderr, "\nLibrary error %d, exiting...\n", r);
	if (file != stdout)
		fclose(file);

	if(!read_iq)
	{
		rtlsdr_close(dev);
	}

	fprintf(stderr,"\nBurst Counter %d",d_counter_bursts);
	free (buffer);
out1:
	return r >= 0 ? r : -r;
}
