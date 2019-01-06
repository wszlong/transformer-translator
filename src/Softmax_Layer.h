
#include <fstream>
#include "transfer_layer.h"

using namespace std;

class Softmax_Layer {
	public:
		
		gpu_info_struct gpu_info;
		int gpu_num;

		int Embedding_size = 1;
		int LSTM_size =1;
		int minibatch_size;
		int vocab_size;
		int Hidden_size; //!
		int VV;
		
		string a_test;
		
		float *d_output_bias;
		float *d_W_c_p1;
		float *d_W_c_p2;

		float *d_final_temp_1;
		float *d_final_temp_2;

		float *d_D; //!!
		float *d_single_x_t; //!!
		float *d_b_d;

		//for small vocab
		float *d_D_small;
		int *d_tgt_vocab;

		//node
		float *d_h_t_source;
		float *d_h_t_below;
		
		float *d_alignment;
		float *d_normal_alignment;
		float *d_c_att_t;

		float *d_ones;
		float *d_outputdist;
		float *d_outputdist_sum;
		float *d_logit_softmax;

		
		lower_transfer_layer lower_layer;		

		Softmax_Layer() {};

		void init_Softmax_Layer(int Hidden_size, int minibatch_size, int vocab_size, int gpu_num);

		void init_params();
		void load_weight(ifstream &input);
		void init_small_vocab(vector<int> tgt_vocab); //

		void softmax_forward_prop(int index, int B, bool is_small_vocab);

};
