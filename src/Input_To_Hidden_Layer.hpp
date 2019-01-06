
//#include "Input_To_Hidden_Layer.h"

void Input_To_Hidden_Layer::init_Input_To_Hidden_Layer(int Embedding_size, int LSTM_size, int minibatch_size, int vocab_size, bool feed_input, int gpu_num, int num_heads, bool bi_dir, Decoder *m) {
	
	this->Embedding_size = Embedding_size;
	this->Hidden_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->vocab_size = vocab_size;
	this->feed_input = feed_input;
	this->gpu_num = gpu_num;
	this->num_heads = num_heads;
	this->is_decoder = bi_dir;
	model = m;
	
	gpu_info.init(gpu_num);
	init_params();
	
	int longest_sent = 200;
	//int num_heads = 8;
	//nodes.clear();
	h_total_x_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_wx_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_c_t = (float **)malloc(longest_sent*sizeof(float*));
	h_alignments = (float **)malloc(minibatch_size*longest_sent*sizeof(float*));
	h_normal_alignments = (float **)malloc(minibatch_size*longest_sent*sizeof(float*));
	
	h_total_h_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_norm_t = (float **)malloc(longest_sent*sizeof(float*)); //h_total_norm_t
	h_total_relu_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_feed_t = (float **)malloc(longest_sent*sizeof(float*)); //feed 
	h_total_norm1_t = (float **)malloc(longest_sent*sizeof(float*)); //
	//h_total_results = (float **)malloc(longest_sent*sizeof(float*)); //norm1

	if (is_decoder) {
		h_total_hdec_t = (float **)malloc(longest_sent*sizeof(float*));
		h_total_norm2_t = (float **)malloc(longest_sent*sizeof(float*)); //
		
		//for encdec att	
		h_total_source_h_t = (float **)malloc(longest_sent*sizeof(float*)); //
		h_total_wx_kvt = (float **)malloc(longest_sent*sizeof(float*)); //
		
		//for decoder update
		h_total_wx_tmp_t = (float **)malloc(longest_sent*sizeof(float*));
	
	}

	for(int i=0; i<longest_sent; i++) {
		
		////nodes.push_back(IH_Node(Hidden_size,minibatch_size,vocab_size,this,i));
		//h_total_x_t[i] = nodes[i].d_x_t;
		
		cudaMalloc((void**) &h_total_x_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_wx_t[i], 3*Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_c_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_h_t[i], Hidden_size*minibatch_size*sizeof(float));
		//cudaMalloc((void**) &h_alignments[i], minibatch_size*num_heads*longest_sent*sizeof(float));
		//cudaMalloc((void**) &h_normal_alignments[i], minibatch_size*num_heads*longest_sent*sizeof(float));
		cudaMalloc((void**) &h_total_norm_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_relu_t[i], 4*Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_feed_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_norm1_t[i], Hidden_size*minibatch_size*sizeof(float));
		
		//h_total_h_t[i] = nodes[i].d_h_t;

		if(is_decoder) {

			cudaMalloc((void**) &h_total_hdec_t[i], Hidden_size*minibatch_size*sizeof(float));
			cudaMalloc((void**) &h_total_norm2_t[i], Hidden_size*minibatch_size*sizeof(float));
			
			cudaMalloc((void**) &h_total_source_h_t[i], Hidden_size*minibatch_size*sizeof(float));
			cudaMalloc((void**) &h_total_wx_kvt[i], 2*Hidden_size*minibatch_size*sizeof(float));
		
			cudaMalloc((void**) &h_total_wx_tmp_t[i], 3*Hidden_size*minibatch_size*sizeof(float)); //for decoder update
		}
	}

	for(int i=0; i<minibatch_size*longest_sent; i++) {
		cudaMalloc((void**) &h_alignments[i], num_heads*longest_sent*sizeof(float));
		cudaMalloc((void**) &h_normal_alignments[i], num_heads*longest_sent*sizeof(float));
	}
	
	cudaMalloc((void**) &d_total_x_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_x_t,h_total_x_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_wx_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_wx_t,h_total_wx_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	cudaMalloc((void**) &d_total_c_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_c_t,h_total_c_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	//alignment weights	
	cudaMalloc((void**) &d_alignments, num_heads*longest_sent*sizeof(float*));
	cudaMemcpy(d_alignments,h_alignments,num_heads*longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_normal_alignments, num_heads*longest_sent*sizeof(float*));
	cudaMemcpy(d_normal_alignments,h_normal_alignments,num_heads*longest_sent*sizeof(float*),cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_total_h_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_h_t,h_total_h_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_norm_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_norm_t,h_total_norm_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	//feed-forward 
	cudaMalloc((void**) &d_total_relu_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_relu_t,h_total_relu_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_feed_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_feed_t,h_total_feed_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_norm1_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_norm1_t,h_total_norm1_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	//cudaMalloc((void**) &d_total_results, longest_sent*sizeof(float*));
	//cudaMemcpy(d_total_results,h_total_results,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	//for decoder
	if (is_decoder){
		cudaMalloc((void**) &d_total_hdec_t, longest_sent*sizeof(float*));
		cudaMemcpy(d_total_hdec_t,h_total_hdec_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
		cudaMalloc((void**) &d_total_norm2_t, longest_sent*sizeof(float*));
		cudaMemcpy(d_total_norm2_t,h_total_norm2_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
		
		cudaMalloc((void**) &d_single_wx_qt, Hidden_size*minibatch_size*sizeof(float));
		
		//for enc-dec att	
		cudaMalloc((void**) &d_total_source_h_t, longest_sent*sizeof(float*));
		cudaMemcpy(d_total_source_h_t,h_total_source_h_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
		cudaMalloc((void**) &d_total_wx_kvt, longest_sent*sizeof(float*));
		cudaMemcpy(d_total_wx_kvt,h_total_wx_kvt,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
		
		//for decoder update
		cudaMalloc((void**) &d_total_wx_tmp_t, longest_sent*sizeof(float*));
		cudaMemcpy(d_total_wx_tmp_t,h_total_wx_tmp_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	}
}

void Input_To_Hidden_Layer::load_weight(ifstream &input) {
	
	cudaSetDevice(gpu_num);
	
	//cout<<"Input_To_Hidden_Layer test load_weight"<<endl;

	//multi-attention
	read_matrix_GPU(d_qkv_kernel,Hidden_size,3*Hidden_size,input);
	read_matrix_GPU(d_qkv_bias,3*Hidden_size,1,input);
	read_matrix_GPU(d_output_transform_kernel,Hidden_size,Hidden_size,input);
	read_matrix_GPU(d_output_transform_bias,Hidden_size,1,input);

	//cout<<"show d_qkv_kernel test"<<endl;
	//show_matrix(d_qkv_kernel,Hidden_size,3*Hidden_size);

	//layer_norm	
	read_matrix_GPU(d_layer_norm_scale,Hidden_size,1,input);
	read_matrix_GPU(d_layer_norm_bias,Hidden_size,1,input);
	read_matrix_GPU(d_layer_norm_1_scale,Hidden_size,1,input);
	read_matrix_GPU(d_layer_norm_1_bias,Hidden_size,1,input);

	//transformer_ffn_layer
	read_matrix_GPU(d_ffn_1_kernel,Hidden_size,4*Hidden_size,input);
	read_matrix_GPU(d_ffn_1_bias,4*Hidden_size,1,input);
	read_matrix_GPU(d_ffn_2_kernel,4*Hidden_size,Hidden_size,input);
	read_matrix_GPU(d_ffn_2_bias,Hidden_size,1,input);
	
	//cout<<"show d_ffn_2_bias"<<endl;
	//show_matrix(d_ffn_2_bias,Hidden_size,1);

	if(is_decoder) {
		
		//decoder inter-attention	
		read_matrix_GPU(d_q_kernel,Hidden_size,Hidden_size,input);
		read_matrix_GPU(d_q_bias,Hidden_size,1,input);
		read_matrix_GPU(d_kv_kernel,Hidden_size,2*Hidden_size,input);
		read_matrix_GPU(d_kv_bias,2*Hidden_size,1,input);

		read_matrix_GPU(d_output_transform_1_kernel,Hidden_size,Hidden_size,input);
		read_matrix_GPU(d_output_transform_1_bias,Hidden_size,1,input);
		
		read_matrix_GPU(d_layer_norm_2_scale,Hidden_size,1,input);
		read_matrix_GPU(d_layer_norm_2_bias,Hidden_size,1,input);
	}
	else { //encoder load d_W
		
		//read_matrix_GPU(d_W,vocab_size,Hidden_size,input); 
		read_matrix_GPU(d_W,Hidden_size,vocab_size,input); //trans
		//read_matrix_GPU(d_target_embedding,32,Hidden_size,input); //target space embedding remove!!!

		//cout<<"show d_W"<<endl;
		//show_matrix_tmp(d_W,Hidden_size,vocab_size);
		//show_matrix(d_W,Hidden_size,vocab_size);
		//show_matrix(d_target_embedding,32,Hidden_size);
	}	
		
	//cout<<"show d_M_i: "<<endl;
	//show_matrix(d_M_i,LSTM_size,Embedding_size);
	
}


void Input_To_Hidden_Layer::init_params() {
	

	cudaSetDevice(gpu_num);
	
	//cout<<"Input_To_Hidden_Layer test init_params"<<endl;
	
	//cudaMalloc((void**) &d_W, Hidden_size*vocab_size*sizeof(float)); //
	
	//multi-attention
	cudaMalloc((void**) &d_qkv_kernel, Hidden_size*3*Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_qkv_bias, 3*Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_output_transform_kernel, Hidden_size*Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_output_transform_bias, Hidden_size*sizeof(float));

	//Residual connection + layer_norm
	cudaMalloc((void**) &d_layer_norm_scale, Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_layer_norm_bias, Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_layer_norm_1_scale, Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_layer_norm_1_bias, Hidden_size*sizeof(float));

	//transformer_ffn_layer
	cudaMalloc((void**) &d_ffn_1_kernel, Hidden_size*4*Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_ffn_1_bias, 4*Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_ffn_2_kernel, 4*Hidden_size*Hidden_size*sizeof(float));
	cudaMalloc((void**) &d_ffn_2_bias, Hidden_size*sizeof(float));
	
	if(is_decoder) {

		//multi-attention  for enc-dec attention
		cudaMalloc((void**) &d_q_kernel, Hidden_size*Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_q_bias, Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_kv_kernel, Hidden_size*2*Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_kv_bias, 2*Hidden_size*sizeof(float));
		
		cudaMalloc((void**) &d_output_transform_1_kernel, Hidden_size*Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_output_transform_1_bias, Hidden_size*sizeof(float));

		//layer_norm
		cudaMalloc((void**) &d_layer_norm_2_scale, Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_layer_norm_2_bias, Hidden_size*sizeof(float));
		
		//look-up table copy from encoder's d_W
		cudaMalloc((void**) &d_W, Hidden_size*vocab_size*sizeof(float));

	}
	else { //encoder 
		
		//look-up table
		cudaMalloc((void**) &d_W, Hidden_size*vocab_size*sizeof(float));
		//cudaMalloc((void**) &d_target_embedding, 32*Hidden_size*sizeof(float));
		
	}
	
	//small vocab
	cudaMalloc((void**)&d_W_dec_small, 10000*Hidden_size*sizeof(float));  // Tmax*50=10000
	cudaMalloc((void**)&d_tgt_vocab, 10000*sizeof(int));
	

	cudaMalloc((void**)&d_temp_1, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_2, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_3, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_4, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_5, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_6, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_7, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_8, LSTM_size*minibatch_size*sizeof(float));


	//node
	cudaMalloc((void**)&d_wid, minibatch_size*200*sizeof(int)); //200 for longest_sent
	cudaMalloc((void**)&d_x_t, Hidden_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**)&d_i_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_f_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_prime_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_o_t, LSTM_size*minibatch_size*sizeof(float));

	cudaMalloc((void**)&d_init_hidden_vector, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_init_cell_vector, LSTM_size*minibatch_size*sizeof(float));
	cudaMemset(d_init_hidden_vector,0,LSTM_size*minibatch_size*sizeof(float));
	cudaMemset(d_init_cell_vector,0,LSTM_size*minibatch_size*sizeof(float));

	cudaMalloc((void**)&d_h_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_h_t_prev, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_t_prev, LSTM_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**)&d_h_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float)); // for prepare_forward_decode
	cudaMalloc((void**)&d_c_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float)); //
	
	cudaMalloc((void**)&d_h_t_feed, Embedding_size*minibatch_size*sizeof(float));

	cudaMalloc((void**)&d_father_idx, longest_sent*minibatch_size*sizeof(int));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Input_To_Hidden_Layer::look_up_gpu(int *input_wids, int Sentence_len) {
	
	//int Sentence_len_src = Sentence_len;	
		
	//minibatch_size = 1 for encoder	
	cudaSetDevice(gpu_num);
	//cudaDeviceSynchronize();
	
	dim3 block_shape(256,1,1);
	//dim3 grid_shape(minibatch_size, (Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	dim3 grid_shape(256, (Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	//int Sentence_len = *input_wids.size();
	cudaMemcpy(d_wid, input_wids, minibatch_size*Sentence_len*sizeof(int), cudaMemcpyHostToDevice);
	
		
	//cout<<"minibatch_size: "<<minibatch_size<<endl;
	//cout<<"Sentence_len: "<<Sentence_len<<endl;
	
	//cout<<"d_W: "<<endl;
	//show_matrix(d_W, Hidden_size, vocab_size);
	
	//cout<<"d_target_embedding: "<<endl;
	//show_matrix(d_target_embedding, 32,Hidden_size);
	
	//cout<<"d_wid: "<<endl;
	//show_matrix(d_wid, Sentence_len, 1);
	
	//cout<<"show nodes[0]: pre "<<endl;
	//show_matrix(h_total_x_t[0], Hidden_size, minibatch_size);
	

	lookup_pa_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_x_t, d_W, d_wid, Hidden_size, Sentence_len, minibatch_size);
	
	//cout<<"Hidden_size: "<<Hidden_size<<endl;	
	//cout<<"show nodes[0]: now "<<endl;
	//show_matrix(h_total_x_t[0], Hidden_size, minibatch_size);
	//cout<<"show nodes[Sentence_len-1]: now "<<endl;
	//show_matrix(h_total_x_t[Sentence_len-1], Hidden_size, minibatch_size);
	
	//remove target embedding!!
	//target_space_and_pos_encoding_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_x_t, d_target_embedding, 9, Hidden_size, Sentence_len, minibatch_size); //target_space:9
	position_encoding_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_x_t, Hidden_size, Sentence_len, minibatch_size); //target_space:9
	CUDA_GET_LAST_ERROR("target_space_and_pos_encoding_kernel failed");
	
	//dim3 grid_shape2(minibatch_size*Sentence_len, (Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	//lookup_pa_kernel_2<<<256,256,0,gpu_info.s0>>>(d_total_x_t, d_W, d_wid, Hidden_size, Sentence_len, minibatch_size);
	
	//dim3 grid_shape2(minibatch_size, (Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	//lookup_kernel<<<grid_shape2,block_shape,0,gpu_info.s0>>>(h_total_x_t[0], d_W, d_wid, Hidden_size);

	//cudaDeviceSynchronize();
	
/*
	cout<<"show nodes[0]: target_space_and_pos_encoding "<<endl;
	show_matrix(h_total_x_t[0], Hidden_size, minibatch_size);
	cout<<"show nodes[Sentence_len-1]: target_space_and_pos_encoding "<<endl;
	show_matrix(h_total_x_t[Sentence_len-1], Hidden_size, minibatch_size);
*/
	//cout<<"show nodes[1]: now "<<endl;
	//show_matrix(nodes[1].d_x_t, Hidden_size, minibatch_size);
	

	//cin>>a_test;
}


void Input_To_Hidden_Layer::init_small_vocab(vector<int> tgt_vocab) {
	
	cudaSetDevice(gpu_num);
	int VV = tgt_vocab.size();
	//cout<<"LSTM_size: "<<LSTM_size<<endl;
	cudaMemcpy(d_tgt_vocab, &tgt_vocab[0], VV*sizeof(int), cudaMemcpyHostToDevice);
	dim3 block_shape0(256,1,1);
	dim3 grid_shape0(VV,(Hidden_size + block_shape0.x - 1)/block_shape0.x,1);
	lookup_kernel<<<grid_shape0,block_shape0,0,gpu_info.s0>>>(d_W_dec_small,d_W,d_tgt_vocab,Hidden_size);
		
	//cout<<"d_W: "<<endl;
	//show_matrix(d_W,Embedding_size,vocab_size);
	//cout<<"d_W_dec_small: "<<endl;
	//show_matrix(d_W_dec_small,Embedding_size,VV);

	CUDA_GET_LAST_ERROR("Input_To_Hidden_Layer::init_small_vocab");	
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Input_To_Hidden_Layer::look_up_gpu_decoder(int *input_wids, int index, int B, bool is_small_vocab) {
	
	cudaSetDevice(gpu_num);
	//cudaDeviceSynchronize();
	
	CUDA_GET_LAST_ERROR("cudaMemcpy(pre) for decoder failed");
	
	dim3 block_shape(256,1,1);
	dim3 grid_shape(B, (Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	cudaMemcpy(d_wid, input_wids, B*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("cudaMemcpy for decoder failed");
/*	
	cout<<"d_W: "<<endl;
	show_matrix(d_W, Hidden_size, vocab_size);
	
	//cout<<"d_wid: "<<endl;
	//show_matrix(d_wid, B, 1);
	cout<<"B: "<<B<<endl;
	cout<<"index: "<<index<<endl;
	cout<<"decoder d_wid: "<<endl;
	show_matrix(d_wid, B, 1);
*/	
	//lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(h_total_x_t[index], d_W, d_wid, Hidden_size);
	//lookup_and_pos_encoding_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(h_total_x_t[index], d_W, d_wid, Hidden_size, index);
	if (is_small_vocab) {
		lookup_and_pos_encoding_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(h_total_x_t[index], d_W_dec_small, d_wid, Hidden_size, index);
	}
	else {
		lookup_and_pos_encoding_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(h_total_x_t[index], d_W, d_wid, Hidden_size, index);
	}
	CUDA_GET_LAST_ERROR("lookup_kernel for decoder failed");
	

	//lookup_pa_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_x_t, d_W, d_wid, Hidden_size, Sentence_len, minibatch_size);
/*	
	cout<<"show decoder h_total_x_t[index]: now "<<endl;
	show_matrix(h_total_x_t[index], Hidden_size, B);
*/	

	//cin>>a_test;
}

void Input_To_Hidden_Layer::update_history_state(int *father_idx, int index, int B) {
	
	cudaSetDevice(gpu_num);

	///cout<<"show update_history_state d_total_wx_t[index-1] pre"<<endl;
	///show_matrix(h_total_wx_t[index-1], 3*Hidden_size, B);
	///cout<<"B: "<<B<<endl;	
	///cout<<"index: "<<index<<endl;	
	
	//cudaMemset(h_total_wx_tmp_t[index-1],0,3*Hidden_size*minibatch_size*sizeof(float));
	
	cudaMemcpy(d_father_idx, father_idx, B*sizeof(int), cudaMemcpyHostToDevice);
	
	///cout<<"show update_history_state d_father_idx: "<<endl;
	///show_matrix(d_father_idx, B*index, 1);
	
	dim3 block_shape(256,1,1);
	dim3 grid_shape(256, (3*Hidden_size + block_shape.x -1)/block_shape.x, 1);  //3*Hidden_size
	update_history_state_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_wx_tmp_t, d_total_wx_t, d_father_idx, 3*Hidden_size, index, B);  // 3*Hidden_size !!
	CUDA_GET_LAST_ERROR("update_history_state_kernel failed");
	//update_history_state_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_wx_tmp_t, d_total_wx_t, d_father_idx, Hidden_size, 1, 4);
	//update_history_state_2_kernel<<<4,256,0,gpu_info.s0>>>(h_total_wx_tmp_t[0], h_total_wx_t[0], d_father_idx, Hidden_size, 1, 4);
	
	for(int i=0; i<index; i++) {
		this->h_total_wx_t[i] = h_total_wx_tmp_t[i];
	}	
	cudaMemcpy(d_total_wx_t,h_total_wx_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice); //!!
	//this->d_total_wx_t = d_total_wx_tmp_t;	
	
	///cout<<"show update_history_state d_total_wx_t[index-1]"<<endl;
	///show_matrix(h_total_wx_t[index-1], 3*Hidden_size, B);
}

void Input_To_Hidden_Layer::multi_head_attention(int Sentence_len) {

	cudaSetDevice(gpu_num);
	
	//cout<<"show d_total_wx_t[0] pre "<<endl;
	//show_matrix(h_total_wx_t[0], 3*Hidden_size, minibatch_size);
	//cout<<"show d_qkv_kernel "<<endl;
	//show_matrix(d_qkv_kernel, Hidden_size, 3*Hidden_size);
	
	dim3 blockPerGrid(256,256);
	calucate_3hidden_matrix_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_wx_t,d_qkv_kernel,d_total_x_t,d_qkv_bias,Hidden_size, Sentence_len, minibatch_size); //right
	
	//cout<<"show d_total_wx_t[0] now "<<endl;
	//show_matrix(h_total_wx_t[0], 3*Hidden_size, minibatch_size);
	//cout<<"show d_total_wx_t[Sentence_len-1] now "<<endl;
	//show_matrix(h_total_wx_t[Sentence_len-1], 3*Hidden_size, minibatch_size);
	
	//cout<<"show d_total_wx_t[0] now value"<<endl;
	//show_matrix_lz(h_total_wx_t[0], 3*Hidden_size, minibatch_size);


	//int num_heads = 8;	
	dim3 block_shape(num_heads,64,1);
	//dim3 block_shape(64,1,1);
	dim3 grid_shape(Sentence_len, 128, 1); //here minibatch_size = 1;
	
	//multi_head_attention_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_h_t, d_total_x_t, d_qkv_kernel, d_qkv_bias, d_output_transform_kernel, d_output_transform_bias);
	
	//multi_head_attention_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_h_t_2, d_total_h_t_1, Hidden_size/2, Sentence_len, minibatch_size, num_heads);
	
	//multi_head_attention_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_h_t, d_total_x_t, Hidden_size/2, Sentence_len, minibatch_size, num_heads);
	
	//cout<<"show d_total_results[0] pre "<<endl;
	//show_matrix(h_total_results[0], num_heads*minibatch_size, Sentence_len);
	
	//multi_head_attention_kernel_2<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_results, d_total_h_t, d_total_wx_t, Hidden_size, Sentence_len, minibatch_size, num_heads);
	
	//multi_head_attention_kernel_3<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_alignments, d_total_h_t, d_total_wx_t, Hidden_size, Sentence_len, minibatch_size, num_heads);
	//multi_head_attention_kernel_4<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_alignments, d_total_h_t, d_total_wx_t, Hidden_size, Sentence_len, minibatch_size, num_heads);

	//cout<<"Hidden_size: "<<Hidden_size<<endl;
	//cout<<"num_heads: "<<num_heads<<endl;
	calucate_mhead_att_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_alignments, d_total_wx_t, Hidden_size, Sentence_len, minibatch_size, num_heads); //right

	///cout<<"Sentence_len: "<<Sentence_len<<endl;
	///cout<<"minibatch_size: "<<minibatch_size<<endl;	

	//cout<<"show d_alignments[0] now "<<endl;
	//show_matrix(h_alignments[0], num_heads, Sentence_len);
	//cout<<"show d_alignments[Sentence_len-1] now "<<endl;
	//show_matrix(h_alignments[Sentence_len-1], num_heads, Sentence_len);
	
	//cout<<"show d_total_results[1] now "<<endl;
	//show_matrix(h_total_results[1], minibatch_size, Sentence_len);
	//

	
	//dim3 block_shape1(minibatch_size*num_heads,1,1);
	//dim3 grid_shape1(Sentence_len,1,1);
	//attention_weight_normalization_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_normal_alignments, d_alignments, Hidden_size, Sentence_len, minibatch_size, num_heads);
	
	//dim3 block_shape1(minibatch_size*num_heads,1,1);
	//dim3 grid_shape1(Sentence_len,1,1);
	//attention_weight_normalization_kernel_2<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_normal_alignments, d_alignments, Hidden_size, Sentence_len, minibatch_size, num_heads);
	
	//cout<<"show d_normal_alignments[0] pre "<<endl;
	//show_matrix(h_normal_alignments[0], num_heads, Sentence_len);
	
	dim3 grid_shape1(128,128,1);
	dim3 block_shape1(64,1,1);
	att_weight_normalization_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_normal_alignments, d_alignments, Hidden_size, Sentence_len, minibatch_size, num_heads); //right
	CUDA_GET_LAST_ERROR("att_weight_normalization_kernel failed");

	//cout<<"show d_normal_alignments[0] now "<<endl;
	//show_matrix(h_normal_alignments[0], num_heads, Sentence_len);
	//cout<<"show d_normal_alignments[Sentence_len-1] now "<<endl;
	//show_matrix(h_normal_alignments[Sentence_len-1], num_heads, Sentence_len);
	
	//for(int i=0; i<200; i++) {
	//	cudaMemset(h_total_c_t[i],0,Hidden_size*minibatch_size*sizeof(float));
	//}
	
	//problems if 03.seg.bpe sentence is enough long.
	//dim3 threadPerBlock(256,128,1);
	creat_total_c_t_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_total_c_t, d_total_wx_t, d_normal_alignments, Hidden_size, Sentence_len, minibatch_size, num_heads); //right except for presision
	
	CUDA_GET_LAST_ERROR("creat_total_c_t_kernel failed");
/*
	cout<<"show d_total_c_t[0] now "<<endl;
	show_matrix(h_total_c_t[0], Hidden_size, minibatch_size);
	//cout<<"show d_total_c_t[1] now "<<endl;
	//show_matrix(h_total_c_t[1], Hidden_size, minibatch_size);
	//cout<<"show d_total_c_t[Sentence_len-2] now "<<endl;
	//show_matrix(h_total_c_t[Sentence_len-2], Hidden_size, minibatch_size);
	cout<<"show d_total_c_t[Sentence_len-1] now "<<endl;
	show_matrix(h_total_c_t[Sentence_len-1], Hidden_size, minibatch_size);
*/	
	//
	calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_bias,Hidden_size, Sentence_len, minibatch_size);
	CUDA_GET_LAST_ERROR("calucate_matrix_multi_kernel failed");

/*	
	cout<<"show d_total_h_t[0] now "<<endl;
	show_matrix(h_total_h_t[0], Hidden_size, minibatch_size);
	cout<<"show d_total_h_t[Sentence_len-1] now "<<endl;
	show_matrix(h_total_h_t[Sentence_len-1], Hidden_size, minibatch_size);
*/	
	//cin>>a_test;	
}

void Input_To_Hidden_Layer::multi_head_att_dec(int index, int B) {

	cudaSetDevice(gpu_num);
	float alpha = 1.0;
	float beta = 0.0;
	
	//cout<<"B: "<<B<<endl;
	//cout<<"index: "<<index<<endl;
	//cout<<"show decoder Input_To_Hidden_Layer d_qkv_kernel "<<endl;
	//show_matrix(d_qkv_kernel, Hidden_size, 3*Hidden_size);
	
	//cudaMemset(h_total_wx_t[index],0,3*Hidden_size*minibatch_size*sizeof(float));	

	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,3*Hidden_size,B,Hidden_size,&alpha,d_qkv_kernel,Hidden_size,h_total_x_t[index],Hidden_size,&beta,h_total_wx_t[index],3*Hidden_size),"d_total_wx_t error");
	//cudaEventRecord(gpu_info.e1, gpu_info.s1);
	
	//cout<<"show decoder d_total_wx_t[index] pre "<<endl;
	//show_matrix(h_total_wx_t[index], 3*Hidden_size, B);
	
	dim3 block_shape(256,1,1);
	dim3 grid_shape(B, (3*Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(h_total_wx_t[index], d_qkv_bias, 3*Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
/*
	cout<<"show decoder d_total_wx_t[0] now "<<endl;
	show_matrix(h_total_wx_t[0], 3*Hidden_size, B);
	
	cout<<"show decoder d_total_wx_t[index] now "<<endl;
	show_matrix(h_total_wx_t[index], 3*Hidden_size, B);
*/	
	
	//********************* decoder self-attention *******************/

	//cudaMemset(h_alignments[index],0,num_heads*(index+1)*sizeof(float));	
	//cout<<"show decoder d_alignments[index] pre "<<endl;
	//show_matrix(h_alignments[index], num_heads, (index+1));
/*	
	for(int i=0; i<B; i++) {	
		
		cudaMemset(h_alignments[index*B+i],0,num_heads*(index+1)*sizeof(float));
		cout<<"show decoder d_alignments[index] pre "<<i<<endl;
		show_matrix(h_alignments[index*B+i], num_heads, (index+1));
	}
*/	
	//dim3 blockPerGrid(B,256,0); ///error
	dim3 blockPerGrid(B,256,1);
	calucate_mhead_att_dec_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_alignments, d_total_wx_t, Hidden_size, index, B, num_heads); //right
	//dim3 blockPerGrid(256,256,1);
	//int sent = index + 1;
	//calucate_mhead_att_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_alignments, d_total_wx_t, Hidden_size, sent, B, num_heads); // use encoder kernel
	CUDA_GET_LAST_ERROR("calucate_mhead_att_dec_kernel failed");
	
	
	//calucate_mhead_att_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_alignments, d_total_wx_t, Hidden_size, Sentence_len, minibatch_size, num_heads);
		
	//cout<<"show decoder d_alignments[index*B+B-1] now "<<endl;
	//show_matrix(h_alignments[index*B+B-1], num_heads, (index+1));
/*	
	for(int i=0; i<B; i++) {	
		cout<<"show decoder d_alignments[index] now "<<i<<endl;
		show_matrix(h_alignments[index*B+i], num_heads, (index+1));
	}
*/	
	dim3 grid_shape1(B,128,1);
	dim3 block_shape1(64,1,1);
	att_weight_norm_dec_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_normal_alignments, d_alignments, Hidden_size, index, B, num_heads);
	CUDA_GET_LAST_ERROR("att_weight_norm_dec_kernel failed");
/*
	cout<<"show dec d_normal_alignments[index*B+B-1] now "<<endl;
	show_matrix(h_normal_alignments[index*B+B-1], num_heads, (index+1));

	for(int i=0; i<(index+1); i++) {
		cout<<"show decoder d_total_wx_t[ "<<i<<" ] now "<<endl;
		show_matrix(h_total_wx_t[i], 3*Hidden_size, B);	
	}	
*/
	creat_total_c_t_dec_kernel<<<blockPerGrid,128,0,gpu_info.s0>>>(d_total_c_t, d_total_wx_t, d_normal_alignments, Hidden_size, index, B, num_heads);
	CUDA_GET_LAST_ERROR("creat_total_c_t_dec_kernel failed");
/*	
	cout<<"show d_total_c_t[index] now "<<endl;
	show_matrix(h_total_c_t[index], Hidden_size, B);
*/	
	//calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_kernel,Hidden_size, Sentence_len, minibatch_size);
	//CUDA_GET_LAST_ERROR("calucate_matrix_multi_kernel failed");
	
	//cout<<"dec d_output_transform_kernel: "<<endl;
	//show_matrix(d_output_transform_kernel, Hidden_size, Hidden_size);	
	//cout<<"show d_total_hdec_t[index] pre pre "<<endl;
	//show_matrix(h_total_hdec_t[index], Hidden_size, B);
	
	//CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,3*Hidden_size,B,Hidden_size,&alpha,d_qkv_kernel,Hidden_size,h_total_x_t[index],Hidden_size,&beta,h_total_wx_t[index],3*Hidden_size),"d_total_wx_t error");
	
	//cudaMemset(h_total_hdec_t[index],0,Hidden_size*minibatch_size*sizeof(float));
	
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,Hidden_size,&alpha,d_output_transform_kernel,Hidden_size,h_total_c_t[index],Hidden_size,&beta,h_total_hdec_t[index],Hidden_size),"d_total_hdec_t error");
	
	////cudaMemset(h_total_hdec_t[index],0,Hidden_size*minibatch_size*sizeof(float));
	
	///cout<<"show d_total_hdec_t[index] pre "<<endl;
	///show_matrix(h_total_hdec_t[index], Hidden_size, B);
	
	dim3 block_shape2(256,1,1);
	dim3 grid_shape2(B, (Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(h_total_hdec_t[index], d_output_transform_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
	
	//cout<<"show dec d_total_hdec_t[index] now "<<endl;
	//show_matrix(h_total_hdec_t[index], Hidden_size, B);
		
	//cin>>a_test;	
}

void Input_To_Hidden_Layer::multi_head_att_encdec(int index, int B, int Sentence_len_src) {

	cudaSetDevice(gpu_num);
	float alpha = 1.0;
	float beta = 0.0;
	
	//cout<<"B: "<<B<<endl;
	//cout<<"index: "<<index<<endl;
	//cout<<"show decoder Input_To_Hidden_Layer d_q_kernel "<<endl;
	//show_matrix(d_q_kernel, Hidden_size, Hidden_size);

	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,Hidden_size,&alpha,d_q_kernel,Hidden_size,h_total_norm_t[index],Hidden_size,&beta,d_single_wx_qt,Hidden_size),"d_total_wx_t error");
	//cudaEventRecord(gpu_info.e1, gpu_info.s1);
	
	//cout<<"show decoder d_total_wx_t[0] pre "<<endl;
	//show_matrix(h_total_wx_t[0], 3*Hidden_size, B);
	
	dim3 block_shape(256,1,1);
	dim3 grid_shape(B, (Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_single_wx_qt, d_q_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
	cudaEventRecord(gpu_info.e0, gpu_info.s0);

	///cout<<"show decoder d_single_wx_qt now "<<endl;
	///show_matrix(d_single_wx_qt, Hidden_size, B);
	
	//cout<<"Sentence_len_src: "<<Sentence_len_src<<endl;	
	
	//cout<<"show decoder d_total_source_h_t[0] now "<<endl;
	//show_matrix(h_total_source_h_t[0], Hidden_size, 1);
	///cout<<"show decoder d_total_source_h_t[-1] now "<<endl;
	///show_matrix(h_total_source_h_t[Sentence_len_src-1], Hidden_size, 1);


	
	dim3 blockPerGrid(256,256);
	calucate_2hidden_matrix_encdec_kernel<<<blockPerGrid,256,0,gpu_info.s1>>>(d_total_wx_kvt,d_total_source_h_t,d_kv_kernel,d_kv_bias,Hidden_size,Sentence_len_src, B);
	CUDA_GET_LAST_ERROR("calucate_2hidden_matrix_encdec_kernel failed");
	cudaEventRecord(gpu_info.e1, gpu_info.s1);
	
	///cout<<"show decoder d_total_wx_kvt[-1] now "<<endl;
	///show_matrix(h_total_wx_kvt[Sentence_len_src-1], Hidden_size, B);

	
	//********************* decoder enc-dec attention *******************/
	cudaStreamWaitEvent(gpu_info.s2, gpu_info.e0, 0);
	cudaStreamWaitEvent(gpu_info.s2, gpu_info.e1, 0);
	dim3 blockPerGrid1(B,256,1);
	calucate_mhead_att_encdec_kernel<<<blockPerGrid1,64,0,gpu_info.s2>>>(d_alignments, d_single_wx_qt, d_total_wx_kvt, Hidden_size, Sentence_len_src, index, B, num_heads);
/*	
	for(int i=0; i<B; i++) {	
		cout<<"show encdec decoder d_alignments[index] now "<<i<<endl;
		show_matrix(h_alignments[index*B+i], num_heads, Sentence_len_src);
	}
*/
	///cout<<"show decoder encdec d_alignments[index] now "<<endl;
	///show_matrix(h_alignments[index], num_heads, Sentence_len_src);
	
	//cout<<"show enc-dec d_normal_alignments[index] pre "<<endl;
	//show_matrix(h_normal_alignments[index], num_heads, Sentence_len_src);
	
	dim3 grid_shape1(B,128,1);
	dim3 block_shape1(64,1,1);
	att_weight_norm_encdec_kernel<<<grid_shape1,block_shape1,0,gpu_info.s2>>>(d_normal_alignments, d_alignments, Hidden_size, Sentence_len_src, index, B, num_heads);
	CUDA_GET_LAST_ERROR("att_weight_norm_encdec_kernel failed");
	
	///cout<<"show enc-dec d_normal_alignments[index] now "<<endl;
	///show_matrix(h_normal_alignments[index], num_heads, Sentence_len_src);
	
	creat_total_c_t_encdec_kernel<<<blockPerGrid1,128,0,gpu_info.s2>>>(d_total_c_t, d_total_wx_kvt, d_normal_alignments, Hidden_size, Sentence_len_src, index, B, num_heads);
	CUDA_GET_LAST_ERROR("creat_total_c_t_encdec_kernel failed");
	
	///cout<<"show d_total_c_t[index] now "<<endl;
	///show_matrix(h_total_c_t[index], Hidden_size, B);
	
	cublasSetStream(gpu_info.handle, gpu_info.s2);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,Hidden_size,&alpha,d_output_transform_1_kernel,Hidden_size,h_total_c_t[index],Hidden_size,&beta,h_total_h_t[index],Hidden_size),"d_total_h_t error");
	
	////cudaMemset(h_total_h_t[index],0,Hidden_size*minibatch_size*sizeof(float));

	///cout<<"show d_total_h_t[index] pre "<<endl;
	///show_matrix(h_total_h_t[index], Hidden_size, B);
	
	dim3 block_shape2(256,1,1);
	dim3 grid_shape2(B, (Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape2,block_shape2,0,gpu_info.s2>>>(h_total_h_t[index], d_output_transform_1_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
	
	cudaEventRecord(gpu_info.e2, gpu_info.s2);
/*	
	cout<<"show decoder dec-att and encdec att d_total_h_t[index] now "<<endl;
	show_matrix(h_total_h_t[index], Hidden_size, B);
*/
	//cin>>a_test;	
}


void Input_To_Hidden_Layer::add_and_norm(int Sentence_len) {
	
	cudaSetDevice(gpu_num);
	///cout<<"Sentence_len: "<<Sentence_len<<endl;	
	///cout<<"minibatch_size: "<<minibatch_size<<endl;	

	//cout<<"show d_total_norm_t[0] pre "<<endl;
	//show_matrix(h_total_norm_t[0], Hidden_size, minibatch_size);
	//cout<<"show d_total_norm_t[Sentence_len-1] pre "<<endl;
	//show_matrix(h_total_norm_t[Sentence_len-1], Hidden_size, minibatch_size);
	
	//dim3 blockPerGrid(256,256,1);	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm_t, d_total_h_t, d_total_x_t, d_layer_norm_scale, d_layer_norm_bias, Hidden_size, Sentence_len, minibatch_size); //right

	//cout<<"show d_total_norm_t[0] now "<<endl;
	//show_matrix(h_total_norm_t[0], Hidden_size, minibatch_size);
	//cout<<"show d_total_norm_t[1] now "<<endl;
	//show_matrix(h_total_norm_t[1], Hidden_size, minibatch_size);
	//cout<<"show d_total_norm_t[Sentence_len-1] now "<<endl;
	//show_matrix(h_total_norm_t[Sentence_len-1], Hidden_size, minibatch_size);
	
	//cin>>a_test;
		
}

void Input_To_Hidden_Layer::add_and_norm_dec(int index, int B) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_dec_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm_t, d_total_hdec_t, d_total_x_t, d_layer_norm_scale, d_layer_norm_bias, Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_dec_kernel failed");
/*
	cout<<"show dec d_total_norm_t[index] now "<<endl;
	show_matrix(h_total_norm_t[index], Hidden_size, B);
*/	
	//cin>>a_test;
		
}

void Input_To_Hidden_Layer::add_and_norm_1_dec(int index, int B) {
	
	cudaSetDevice(gpu_num);

	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e2, 0);	

	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_dec_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm1_t, d_total_h_t, d_total_norm_t, d_layer_norm_1_scale, d_layer_norm_1_bias, Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_dec_kernel failed");
/*
	cout<<"show after encdec-att d_total_norm1_t[index] now "<<endl;
	show_matrix(h_total_norm1_t[index], Hidden_size, B);
*/	
	//cin>>a_test;
		
}

void Input_To_Hidden_Layer::add_and_norm_2_dec(int index, int B) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_dec_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm2_t, d_total_feed_t, d_total_norm1_t, d_layer_norm_2_scale, d_layer_norm_2_bias, Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_dec_kernel failed");
/*
	cout<<"show I2H after feed d_total_norm2_t[index] now "<<endl;
	show_matrix(h_total_norm2_t[index], Hidden_size, B);
*/	
	if(upper_layer.copy_h_t) {
			cudaMemcpyAsync(upper_layer.hidden_layer->h_total_x_t[index], h_total_norm2_t[index], Hidden_size*B*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
		//upper_layer.hidden_layer->h_total_x_t[0] = h_total_h3_t[0];
		CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");
			
	}
	else {
		
	}
	
	
	//cin>>a_test;
		
}

void Input_To_Hidden_Layer::feed_foward(int Sentence_len) {

	cudaSetDevice(gpu_num);
	
	//cout<<"show d_ffn_1_kernel now "<<endl;
	//show_matrix(d_ffn_1_kernel, Hidden_size, 4*Hidden_size);
	
	//cout<<"show d_ffn_1_bias now "<<endl;
	//show_matrix(d_ffn_1_bias, 4*Hidden_size,1);
	
	dim3 blockPerGrid(256,256,1);
	calcuate_matrix_multi_and_relu_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_relu_t, d_total_norm_t, d_ffn_1_kernel, d_ffn_1_bias, Hidden_size, Sentence_len, minibatch_size);	
	CUDA_GET_LAST_ERROR("calcuate_matrix_multi_and_relu_kernel failed");
	
	//???too much zero ?	
	//cout<<"show d_total_relu_t[0] now "<<endl;
	//show_matrix(h_total_relu_t[0], Hidden_size, minibatch_size);
	//cout<<"show d_total_relu_t[Sentence_len-1] now "<<endl;
	//show_matrix_test(h_total_relu_t[Sentence_len-1], Hidden_size, minibatch_size);
	
	calcuate_matrix_multi_after_relu_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_feed_t, d_total_relu_t, d_ffn_2_kernel, d_ffn_2_bias, Hidden_size, Sentence_len, minibatch_size);	
	CUDA_GET_LAST_ERROR("calcuate_matrix_multi_after_relu_kernel failed");
	//feed_foward_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h1_t, d_total_h_t, d_total_x_t, d_layer_norm_scale, d_layer_norm_bias, Hidden_size, Sentence_len, minibatch_size);
	
	//cout<<"show d_total_h2_t[0] now "<<endl;
	//show_matrix(h_total_h2_t[0], Hidden_size, minibatch_size);
	//cout<<"show d_total_feed_t[Sentence_len-1] now "<<endl;
	//show_matrix(h_total_feed_t[Sentence_len-1], Hidden_size, minibatch_size);
	
	//cin>>a_test;
		
	//calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_kernel,Hidden_size, Sentence_len, minibatch_size);
	
}


void Input_To_Hidden_Layer::feed_foward_dec(int index, int B) {

	cudaSetDevice(gpu_num);
	
	//cout<<"show d_ffn_1_kernel now "<<endl;
	//show_matrix(d_ffn_1_kernel, Hidden_size, 4*Hidden_size);
	
	//cout<<"show d_ffn_1_bias now "<<endl;
	//show_matrix(d_ffn_1_bias, 4*Hidden_size,1);
	float alpha = 1.0;
	float beta = 0.0;

	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,4*Hidden_size,B,Hidden_size,&alpha,d_ffn_1_kernel,Hidden_size,h_total_norm1_t[index],Hidden_size,&beta,h_total_relu_t[index],4*Hidden_size),"d_total_relu_t error");
	//cudaEventRecord(gpu_info.e1, gpu_info.s1);
	
	
	dim3 block_shape(256,1,1);
	dim3 grid_shape(B, (4*Hidden_size + block_shape.x -1)/block_shape.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(h_total_relu_t[index], d_ffn_1_bias, 4*Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");

	//!!relu
	dim3 block_shape2(256,1,1);
	dim3 grid_shape2(B, (4*Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //4*Hidden_size
	matrix_relu_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(h_total_relu_t[index],4*Hidden_size);
	CUDA_GET_LAST_ERROR("matrix_relu_kernel failed");


	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,4*Hidden_size,&alpha,d_ffn_2_kernel,4*Hidden_size,h_total_relu_t[index],4*Hidden_size,&beta,h_total_feed_t[index],Hidden_size),"d_total_relu_t error");

	//dim3 block_shape2(256,1,1);
	//dim3 grid_shape2(B, (Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(h_total_feed_t[index], d_ffn_2_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
/*
	cout<<"show after enc-dec att d_total_feed_t[index] now "<<endl;
	show_matrix(h_total_feed_t[index], Hidden_size, B);
*/	
	//cin>>a_test;
		
	//calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_kernel,Hidden_size, Sentence_len, minibatch_size);
	

}

void Input_To_Hidden_Layer::add_and_norm_1(int Sentence_len) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm1_t, d_total_feed_t, d_total_norm_t, d_layer_norm_1_scale, d_layer_norm_1_bias, Hidden_size, Sentence_len, minibatch_size);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_kernel failed");

	//cout<<"show d_total_norm1_t[0] now "<<endl;
	//show_matrix(h_total_norm1_t[0], Hidden_size, minibatch_size);
	//cout<<"show d_total_norm1_t[Sentence_len-1] now "<<endl;
	//show_matrix(h_total_norm1_t[Sentence_len-1], Hidden_size, minibatch_size);

	//cout<<"show d_total_x_t[0] upper_layer now "<<endl;
	//show_matrix(upper_layer.hidden_layer->h_total_x_t[0], Hidden_size, minibatch_size);
	//cout<<"show Input_To_Hidden_Layer.hpp d_qkv_kernel, upper_layer now "<<endl;
	//show_matrix(upper_layer.hidden_layer->d_qkv_kernel, Hidden_size, 3*Hidden_size);

	if(upper_layer.copy_h_t) {
		for(int i=0; i<Sentence_len; i++) {	
			cudaMemcpyAsync(upper_layer.hidden_layer->h_total_x_t[i], h_total_norm1_t[i], Hidden_size*minibatch_size*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
		}
		//upper_layer.hidden_layer->h_total_x_t[0] = h_total_h3_t[0];
		CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");
			
	}
	else {
		
	}
	
	cudaEventRecord(gpu_info.h_t_below_transfer, gpu_info.s0);

	//cin>>a_test;
}



