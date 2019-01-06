
#ifndef DECODER_H
#define DECODER_H

//#define IDX2C(i,j,ld) (((j)*(ld))+(i))
using namespace std;

class Input_To_Hidden_Layer;

class Hidden_To_Hidden_Layer;

class Softmax_Layer;

class Decoder {

    public:
        map<string,int> src_w2i;
        map<int,string> tgt_i2w;
		vector<int> father_idx;

        string a_test; //for test
		int B, K, Tmax, T;
		
		bool feed_input = false;
		bool is_small_vocab = false;


		int Hidden_size,Embedding_size, LSTM_size, num_heads, num_layers_src, num_layers_tgt, target_vocab_size, source_vocab_size;
        
		int gpu_num, beam_size;
		string input_weight_file;
		string input_vocab_file;
		string small_vocab_file;

		map<int,vector<int>> s2t;
		vector<int> tgt_vocab;
		
		Decoder(string input_weight_file, string input_vocab_file, string small_vocab_file, int beam_size, int gpu_num);
        string translate(string input_sen);
		
		
		Input_To_Hidden_Layer input_layer_source;
		Input_To_Hidden_Layer input_layer_target;

		vector<Hidden_To_Hidden_Layer> source_hidden_layers;
		vector<Hidden_To_Hidden_Layer> target_hidden_layers;

		Softmax_Layer softmax_layer_target;
    	

	//private:
		void init_and_load_model(map<string,int> &src_w2i,map<int,string> &tgt_i2w);
		
		void load_vocab();		
		void creat_model_structure();
		void init_model_alloc();
		void load_model_params();

        vector<int> w2id(string input_sen);
        string id2w(vector<int> output_wids);
        void encode(vector<int> input_wids);
        vector<int> decode();
        void get_next_prob(vector<int> tgt_wids, int index, vector<int> father_idx);
        void generate_new_samples(vector<vector<int> > &hyp_samples,vector<float> &hyp_scores,
                vector<vector<int> > &final_samples, vector<float> &final_scores, int &dead_k, vector<int> &tgt_wids, vector<int> &father_idx);

		
};

#endif
