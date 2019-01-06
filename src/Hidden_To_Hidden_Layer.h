
#include <fstream>
#include "transfer_layer.h"

using namespace std;

class Decoder;

class Hidden_To_Hidden_Layer {
	public:
		
		//std::vector<HH_Node> nodes;
		Decoder *model;

		gpu_info_struct gpu_info;
		int gpu_num;

		int LSTM_size;
		int Hidden_size; //!!
		bool is_decoder; //!!
		int num_layers_tgt; //!!
		int Sentence_len;
		int longest_sent = 200;
		int num_heads;
		int minibatch_size;
	
		string a_test;	

		//multi-attention
		float *d_qkv_kernel;
		float *d_qkv_bias;
		float *d_output_transform_kernel;
		float *d_output_transform_bias;

		//layer_norm
		float *d_layer_norm_scale;
		float *d_layer_norm_bias;
		float *d_layer_norm_1_scale;
		float *d_layer_norm_1_bias;

		//transformer_ffn_layer
		float *d_ffn_1_kernel;
		float *d_ffn_1_bias;
		float *d_ffn_2_kernel;
		float *d_ffn_2_bias;

		//for decoder inter-attention
		float *d_q_kernel;
		float *d_q_bias;
		float *d_kv_kernel;
		float *d_kv_bias;
		
		float *d_output_transform_1_kernel;
		float *d_output_transform_1_bias;

		float *d_layer_norm_2_scale;
		float *d_layer_norm_2_bias;

		//!!
		float **d_total_x_t;
		float **h_total_x_t;
		float **d_total_wx_t;
		float **h_total_wx_t;
		float **d_total_c_t;
		float **h_total_c_t;
		float **d_total_h_t;
		float **h_total_h_t;
		
		float **d_alignments;
		float **h_alignments;
		float **d_normal_alignments;
		float **h_normal_alignments;
		
		float **d_total_norm_t;
		float **h_total_norm_t;
		
		float **d_total_relu_t;
		float **h_total_relu_t;
		float **d_total_feed_t;
		float **h_total_feed_t;

		float **d_total_norm1_t;
		float **h_total_norm1_t;
		
		//for decoder	
		float **d_total_hdec_t;
		float **h_total_hdec_t;
		float **d_total_norm2_t;
		float **h_total_norm2_t;
		
		float *d_single_wx_qt;
		
		float **d_total_source_h_t;
		float **h_total_source_h_t;
		float **d_total_wx_kvt;
		float **h_total_wx_kvt;

		//for decoder update
		float **d_total_wx_tmp_t;
		float **h_total_wx_tmp_t;

		float *d_temp_1;
		float *d_temp_2;
		float *d_temp_3;
		float *d_temp_4;
		float *d_temp_5;
		float *d_temp_6;
		float *d_temp_7;
		float *d_temp_8;
		
		float *d_temp_1_bi;
		float *d_temp_3_bi;
		float *d_temp_5_bi;
		float *d_temp_7_bi;

		//node
		float *d_i_t;
		float *d_f_t;
		float *d_c_prime_t;
		float *d_o_t;

		float *d_init_hidden_vector;
		float *d_init_cell_vector;
		
		float *d_h_t;
		float *d_c_t;	
		float *d_h_t_prev;
		float *d_c_t_prev;
		float *d_h_t_below;
		float *d_h_t_below_bi;

		
		float *d_h_t_prev_tmp;
		float *d_c_t_prev_tmp;

		int *d_father_idx;

		
		upper_transfer_layer upper_layer;
		lower_transfer_layer lower_layer;
		
		Hidden_To_Hidden_Layer() {};

		void init_Hidden_To_Hidden_Layer(int LSTM_size, int minibatch_size, int gpu_num, int num_heads, bool is_decoder, int num_layers_tgt, struct Decoder *m);

		void init_params();
		void load_weight(ifstream &input);
		void multi_head_attention(int Sentence_len);
		void add_and_norm(int Sentence_len);
		void feed_foward(int Sentence_len);
		void add_and_norm_1(int Sentence_len);
		
		//for decoder
		void multi_head_att_dec(int index, int B);
		void add_and_norm_dec(int index, int B);
		
		void multi_head_att_encdec(int index, int B, int Sentence_len_src);
		void add_and_norm_1_dec(int index, int B);
		void feed_foward_dec(int index, int B);
		void add_and_norm_2_dec(int index, int B);
		
		void forward_prop_sync(cudaStream_t &my_s);
		
		void update_history_state(int *father_wids, int index, int B);
		
		
		void prepare_forward(float * d_h_t_prev, float * d_c_t_prev);
		void forward_prop(int index, int T, int B);

		void prepare_forward_decode(int *father_idx, int B, cudaEvent_t &prev_event);

};
