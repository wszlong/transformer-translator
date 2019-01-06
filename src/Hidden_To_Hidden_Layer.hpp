
//#include "Hidden_To_Hidden_Layer.h"

void Hidden_To_Hidden_Layer::init_Hidden_To_Hidden_Layer(int LSTM_size, int minibatch_size, int gpu_num, int num_heads, bool is_decoder, int num_layers_tgt, Decoder *m) {
	this->Hidden_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->gpu_num = gpu_num;
	this->num_heads = num_heads;
	this->is_decoder = is_decoder; //!
    this->num_layers_tgt = num_layers_tgt;	
	this->model = m;

	gpu_info.init(gpu_num);
	init_params();
	
	int longest_sent = 200;
	//nodes.clear();

	h_total_x_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_wx_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_c_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_h_t = (float **)malloc(longest_sent*sizeof(float*));
	
	h_alignments = (float **)malloc(minibatch_size*longest_sent*sizeof(float*));
	h_normal_alignments = (float **)malloc(minibatch_size*longest_sent*sizeof(float*));

	h_total_norm_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_relu_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_feed_t = (float **)malloc(longest_sent*sizeof(float*));
	h_total_norm1_t = (float **)malloc(longest_sent*sizeof(float*));

	if (is_decoder) {
		h_total_hdec_t = (float **)malloc(longest_sent*sizeof(float*));
		h_total_norm2_t = (float **)malloc(longest_sent*sizeof(float*)); //
		
		//for encdec att	
		h_total_source_h_t = (float **)malloc(longest_sent*sizeof(float*)); //
		h_total_wx_kvt = (float **)malloc(longest_sent*sizeof(float*)); //
	
		//
		h_total_wx_tmp_t = (float **)malloc(longest_sent*sizeof(float*));
	
	}

	for(int i=0; i<longest_sent; i++) {
		//nodes.push_back(HH_Node(Hidden_size,minibatch_size,this,i));

		cudaMalloc((void**) &h_total_x_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_wx_t[i], 3*Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_c_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_h_t[i], Hidden_size*minibatch_size*sizeof(float));
		
		cudaMalloc((void**) &h_total_norm_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_relu_t[i], 4*Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_feed_t[i], Hidden_size*minibatch_size*sizeof(float));
		cudaMalloc((void**) &h_total_norm1_t[i], Hidden_size*minibatch_size*sizeof(float));
		
		if(is_decoder) {

			cudaMalloc((void**) &h_total_hdec_t[i], Hidden_size*minibatch_size*sizeof(float));
			cudaMalloc((void**) &h_total_norm2_t[i], Hidden_size*minibatch_size*sizeof(float));
			
			cudaMalloc((void**) &h_total_source_h_t[i], Hidden_size*minibatch_size*sizeof(float));
			cudaMalloc((void**) &h_total_wx_kvt[i], 2*Hidden_size*minibatch_size*sizeof(float));
		
			//
			cudaMalloc((void**) &h_total_wx_tmp_t[i], 3*Hidden_size*minibatch_size*sizeof(float));
		}

	}
	
	for(int i=0; i<minibatch_size*longest_sent; i++) {
		cudaMalloc((void**) &h_alignments[i], num_heads*longest_sent*sizeof(float));
		cudaMalloc((void**) &h_normal_alignments[i], num_heads*longest_sent*sizeof(float));
	}
	
	//alignment weights	
	cudaMalloc((void**) &d_alignments, num_heads*longest_sent*sizeof(float*));
	cudaMemcpy(d_alignments,h_alignments,num_heads*longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_normal_alignments, num_heads*longest_sent*sizeof(float*));
	cudaMemcpy(d_normal_alignments,h_normal_alignments,num_heads*longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	//multi-head attention
	cudaMalloc((void**) &d_total_x_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_x_t,h_total_x_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_wx_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_wx_t,h_total_wx_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_c_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_c_t,h_total_c_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_h_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_h_t,h_total_h_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
	//norm and feed-forward
	cudaMalloc((void**) &d_total_norm_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_norm_t,h_total_norm_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_relu_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_relu_t,h_total_relu_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_feed_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_feed_t,h_total_feed_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_total_norm1_t, longest_sent*sizeof(float*));
	cudaMemcpy(d_total_norm1_t,h_total_norm1_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	
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
		
		//
		cudaMalloc((void**) &d_total_wx_tmp_t, longest_sent*sizeof(float*));
		cudaMemcpy(d_total_wx_tmp_t,h_total_wx_tmp_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice);
	}

}

void Hidden_To_Hidden_Layer::load_weight(ifstream &input) {
	
	cudaSetDevice(gpu_num);
	//cout<<"Hidden_To_Hidden_Layer test load_weight"<<endl;	
	
	//multi-attention
	read_matrix_GPU(d_qkv_kernel,Hidden_size,3*Hidden_size,input);
	read_matrix_GPU(d_qkv_bias,3*Hidden_size,1,input);
	read_matrix_GPU(d_output_transform_kernel,Hidden_size*Hidden_size,1,input);
	read_matrix_GPU(d_output_transform_bias,Hidden_size,1,input);

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

}

void Hidden_To_Hidden_Layer::init_params() {
	
	cudaSetDevice(gpu_num);
	
	//cout<<"Hidden_To_Hidden_Layer test init_params"<<endl;	
	
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
		//multi-attention  for mask self-attention
		cudaMalloc((void**) &d_q_kernel, Hidden_size*Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_q_bias, Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_kv_kernel, Hidden_size*2*Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_kv_bias, 2*Hidden_size*sizeof(float));
		
		cudaMalloc((void**) &d_output_transform_1_kernel, Hidden_size*Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_output_transform_1_bias, Hidden_size*sizeof(float));

		//layer_norm
		cudaMalloc((void**) &d_layer_norm_2_scale, Hidden_size*sizeof(float));
		cudaMalloc((void**) &d_layer_norm_2_bias, Hidden_size*sizeof(float));
	}
	


	cudaMalloc((void**)&d_temp_1, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_2, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_3, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_4, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_5, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_6, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_7, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_8, LSTM_size*minibatch_size*sizeof(float));
	
	if(lower_layer.lower_input && upper_layer.source_side) {
		cudaMalloc((void**)&d_temp_1_bi, LSTM_size*minibatch_size*sizeof(float));
		cudaMalloc((void**)&d_temp_3_bi, LSTM_size*minibatch_size*sizeof(float));
		cudaMalloc((void**)&d_temp_5_bi, LSTM_size*minibatch_size*sizeof(float));
		cudaMalloc((void**)&d_temp_7_bi, LSTM_size*minibatch_size*sizeof(float));
	}

	//node
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
	
	cudaMalloc((void**)&d_h_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**)&d_h_t_below, LSTM_size*minibatch_size*sizeof(float));
	
	//
	cudaMalloc((void**)&d_h_t_below_bi, LSTM_size*200*sizeof(float));

	cudaMalloc((void**)&d_father_idx, minibatch_size*sizeof(int));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}


void Hidden_To_Hidden_Layer::forward_prop_sync(cudaStream_t &my_s) {
	
	if(lower_layer.lower_input) {
		cudaStreamWaitEvent(my_s, lower_layer.input_layer->gpu_info.h_t_below_transfer, 0);
	}
	else {
		cudaStreamWaitEvent(my_s, lower_layer.hidden_layer->gpu_info.h_t_below_transfer, 0);
	}
}

void Hidden_To_Hidden_Layer::update_history_state(int *father_idx, int index, int B) {
	
	cudaSetDevice(gpu_num);
	
	cudaMemcpy(d_father_idx, father_idx, B*sizeof(int), cudaMemcpyHostToDevice);
	
	CUDA_GET_LAST_ERROR("update_history_state_kernel H2H(pre) failed");	
	
	dim3 block_shape(256,1,1);
	dim3 grid_shape(256, (3*Hidden_size + block_shape.x -1)/block_shape.x, 1); //here 3*Hidden_size!!!
	/*
	cout<<"show update_history_state d_father_idx: "<<endl;
	show_matrix(d_father_idx, B, 1);
	
	cout<<"show update_history_state d_total_wx_t[index-1]"<<endl;
	show_matrix(h_total_wx_t[index-1], 3*Hidden_size, B);
	*/

	update_history_state_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_total_wx_tmp_t, d_total_wx_t, d_father_idx, 3*Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("update_history_state_kernel H2H failed");	
	//this->d_total_wx_t = d_total_wx_tmp_t;	
	for(int i=0; i<index; i++) {
		this->h_total_wx_t[i] = h_total_wx_tmp_t[i];
	}	
	cudaMemcpy(d_total_wx_t,h_total_wx_t,longest_sent*sizeof(float*),cudaMemcpyHostToDevice); //!!
	
	//cout<<"show update_history_state d_total_wx_t[index-1]"<<endl;
	//show_matrix(h_total_wx_t[index-1], 3*Hidden_size, B);
}


void Hidden_To_Hidden_Layer::multi_head_attention(int Sentence_len) {
	
	cudaSetDevice(gpu_num);
/*
	cout<<"show Hidden_To_Hidden_Layer d_total_x_t[0] now "<<endl;
	show_matrix(h_total_x_t[0], Hidden_size, minibatch_size);
	cout<<"show Hidden_To_Hidden_Layer d_total_x_t[Sentence_len-1] now "<<endl;
	show_matrix(h_total_x_t[Sentence_len-1], Hidden_size, minibatch_size);
*/	
	//cout<<"show d_total_x_t[0] lower_layer now "<<endl;
	//show_matrix(lower_layer.input_layer->h_total_x_t[0], Hidden_size, minibatch_size);
	
	forward_prop_sync(gpu_info.s0);
		
	//cout<<"show Hidden_To_Hidden_Layer d_total_x_t[0] now "<<endl;
	//show_matrix(h_total_x_t[0], Hidden_size, minibatch_size);
	//cout<<"show Hidden_To_Hidden_Layer d_q_kernel now "<<endl;
	//show_matrix(d_qkv_kernel, Hidden_size, 3*Hidden_size);
	
	//cout<<"show Hidden_To_Hidden_Layer d_total_wx_t[0] pre "<<endl;
	//show_matrix(h_total_wx_t[0], 3*Hidden_size, minibatch_size);
	
	dim3 blockPerGrid(256,256);
	calucate_3hidden_matrix_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_wx_t,d_qkv_kernel,d_total_x_t,d_qkv_bias,Hidden_size, Sentence_len, minibatch_size);
	CUDA_GET_LAST_ERROR("calucate_3hidden_matrix_kernel failed");
	
	//cout<<"Sentence_len: "<<Sentence_len<<endl;
	//cout<<"minibatch_size: "<<minibatch_size<<endl;

	//cout<<"show Hidden_To_Hidden_Layer d_total_wx_t[0] now "<<endl;
	//show_matrix_test(h_total_wx_t[0], 3*Hidden_size, minibatch_size);
	//cout<<"show Hidden_To_Hidden_Layer d_total_wx_t[Sentence_len-1] now "<<endl;
	//show_matrix_test(h_total_wx_t[Sentence_len-1], 3*Hidden_size, minibatch_size);
	
	calucate_mhead_att_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_alignments, d_total_wx_t, Hidden_size, Sentence_len, minibatch_size, num_heads);
	CUDA_GET_LAST_ERROR("calucate_mhead_att_kernel failed");
	
	//cout<<"show d_alignments[0] now "<<endl;
	//show_matrix(h_alignments[0], num_heads, Sentence_len);
	//cout<<"show d_alignments[Sentence_len-1] now "<<endl;
	//show_matrix(h_alignments[Sentence_len-1], num_heads, Sentence_len);
	
	dim3 grid_shape1(128,128,1);
	dim3 block_shape1(64,1,1);
	att_weight_normalization_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_normal_alignments, d_alignments, Hidden_size, Sentence_len, minibatch_size, num_heads);
	CUDA_GET_LAST_ERROR("att_weight_normalization_kernel failed");
	
	//cout<<"show d_normal_alignments[0] now "<<endl;
	//show_matrix(h_normal_alignments[0], num_heads, Sentence_len);
	//cout<<"show d_normal_alignments[Sentence_len-1] now "<<endl;
	//show_matrix(h_normal_alignments[Sentence_len-1], num_heads, Sentence_len);

	creat_total_c_t_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_total_c_t, d_total_wx_t, d_normal_alignments, Hidden_size, Sentence_len, minibatch_size, num_heads);
	CUDA_GET_LAST_ERROR("creat_total_c_t_kernel failed");
	
	//cout<<"show Hidden_To_Hidden_Layer d_total_c_t[0] now "<<endl;
	//show_matrix(h_total_c_t[0], Hidden_size, minibatch_size);

	calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_bias,Hidden_size, Sentence_len, minibatch_size); //kernel bias
	CUDA_GET_LAST_ERROR("calucate_matrix_multi_kernel failed");
	
	//cout<<"show Hidden_To_Hidden_Layer d_total_h_t[0] now "<<endl;
	//show_matrix(h_total_h_t[0], Hidden_size, minibatch_size);
	//cout<<"show Hidden_To_Hidden_Layer d_total_h_t[Sentence_len-1] now "<<endl;
	//show_matrix(h_total_h_t[Sentence_len-1], Hidden_size, minibatch_size);

	//cin>>a_test;

}

void Hidden_To_Hidden_Layer::add_and_norm(int Sentence_len) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm_t, d_total_h_t, d_total_x_t, d_layer_norm_scale, d_layer_norm_bias, Hidden_size, Sentence_len, minibatch_size);

	//cout<<"show d_total_h1_t[0] now "<<endl;
	//show_matrix(h_total_h1_t[0], Hidden_size, minibatch_size);
	
	//cout<<"show d_total_h1_t[2] now "<<endl;
	//show_matrix(h_total_h1_t[2], Hidden_size, minibatch_size);
	
	//cin>>a_test;
		
}

void Hidden_To_Hidden_Layer::feed_foward(int Sentence_len) {

	cudaSetDevice(gpu_num);
	
	//cout<<"show d_ffn_1_kernel now "<<endl;
	//show_matrix(d_ffn_1_kernel, Hidden_size, 4*Hidden_size);
	
	//cout<<"show d_ffn_1_bias now "<<endl;
	//show_matrix(d_ffn_1_bias, 4*Hidden_size,1);
	
	dim3 blockPerGrid(256,256,1);
	calcuate_matrix_multi_and_relu_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_relu_t, d_total_norm_t, d_ffn_1_kernel, d_ffn_1_bias, Hidden_size, Sentence_len, minibatch_size);	
	CUDA_GET_LAST_ERROR("calcuate_matrix_multi_and_relu_kernel failed");
	
	//cout<<"show d_total_relu_t[Sentence_len-1] now "<<endl;
	//show_matrix(h_total_relu_t[Sentence_len-1], Hidden_size, minibatch_size);
	
	calcuate_matrix_multi_after_relu_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_feed_t, d_total_relu_t, d_ffn_2_kernel, d_ffn_2_bias, Hidden_size, Sentence_len, minibatch_size);	
	CUDA_GET_LAST_ERROR("calcuate_matrix_multi_after_relu_kernel failed");
	//feed_foward_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h1_t, d_total_h_t, d_total_x_t, d_layer_norm_scale, d_layer_norm_bias, Hidden_size, Sentence_len, minibatch_size);
	
	//cout<<"show d_total_h2_t[0] now "<<endl;
	//show_matrix(h_total_h2_t[0], Hidden_size, minibatch_size);
	
	//cout<<"show d_total_h2_t[2] now "<<endl;
	//show_matrix(h_total_h2_t[2], Hidden_size, minibatch_size);
	
	//cin>>a_test;
		
	//calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_kernel,Hidden_size, Sentence_len, minibatch_size);
	

}

void Hidden_To_Hidden_Layer::add_and_norm_1(int Sentence_len) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm1_t, d_total_feed_t, d_total_norm_t, d_layer_norm_1_scale, d_layer_norm_1_bias, Hidden_size, Sentence_len, minibatch_size);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_kernel failed");
/*
	cout<<"show Hidden_To_Hidden_Layer d_total_norm1_t[0] now "<<endl;
	show_matrix(h_total_norm1_t[0], Hidden_size, minibatch_size);
	cout<<"show Hidden_To_Hidden_Layer d_total_norm1_t[Sentence_len-1] now "<<endl;
	show_matrix(h_total_norm1_t[Sentence_len-1], Hidden_size, minibatch_size);
*/	
	//cout<<"show d_total_h3_t[1] now "<<endl;
	//show_matrix(h_total_h3_t[1], Hidden_size, minibatch_size);
	
	//cout<<"show d_total_x_t[0] upper_layer now "<<endl;
	//show_matrix(upper_layer.hidden_layer->h_total_x_t[0], Hidden_size, minibatch_size);
	//cout<<"show Input_To_Hidden_Layer.hpp d_qkv_kernel, upper_layer now "<<endl;
	//show_matrix(upper_layer.hidden_layer->d_qkv_kernel, Hidden_size, 3*Hidden_size);

	if(upper_layer.copy_h_t && upper_layer.hidden_layer) {
		for(int i=0; i<Sentence_len; i++) {	
			cudaMemcpyAsync(upper_layer.hidden_layer->h_total_x_t[i], h_total_norm1_t[i], Hidden_size*minibatch_size*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
		}
		//upper_layer.hidden_layer->h_total_x_t[0] = h_total_h3_t[0];
		CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");
			
	}
	else if(upper_layer.copy_h_t && upper_layer.softmax && !is_decoder) {
		
		for(int i=0; i<Sentence_len; i++) {	
			cudaMemcpyAsync(model->input_layer_target.h_total_source_h_t[i], h_total_norm1_t[i], Hidden_size*minibatch_size*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
		}
		CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");
		
		//!!	
		for(int i=0; i<num_layers_tgt-1; i++) {
			for(int j=0; j<Sentence_len; j++) {	
				cudaMemcpyAsync(model->target_hidden_layers[i].h_total_source_h_t[j], h_total_norm1_t[j], Hidden_size*minibatch_size*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
			}
			CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");	
		}
		
	}

	else {
		
	}
	
	cudaEventRecord(gpu_info.h_t_below_transfer, gpu_info.s0);

	///cin>>a_test;
}

void Hidden_To_Hidden_Layer::multi_head_att_dec(int index, int B) {

	cudaSetDevice(gpu_num);
	float alpha = 1.0;
	float beta = 0.0;
	
	///cout<<"B: "<<B<<endl;
	///cout<<"index: "<<index<<endl;
	//cout<<"show decoder Input_To_Hidden_Layer d_qkv_kernel "<<endl;
	//show_matrix(d_qkv_kernel, Hidden_size, 3*Hidden_size);
	
	//cout<<"show H2H decoder d_total_x_t[index] "<<endl;
	//show_matrix(h_total_x_t[index], Hidden_size, B);

	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,3*Hidden_size,B,Hidden_size,&alpha,d_qkv_kernel,Hidden_size,h_total_x_t[index],Hidden_size,&beta,h_total_wx_t[index],3*Hidden_size),"d_total_wx_t error");
	//cudaEventRecord(gpu_info.e1, gpu_info.s1);
	
	//cout<<"show decoder d_total_wx_t[0] pre "<<endl;
	//show_matrix(h_total_wx_t[0], 3*Hidden_size, B);
	
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

	
	//cout<<"show decoder d_alignments[index] pre "<<endl;
	//show_matrix(h_alignments[index], num_heads, (index+1));
	//cout<<"B: "<<B<<endl;	
	dim3 blockPerGrid(B,256,1);
	calucate_mhead_att_dec_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_alignments, d_total_wx_t, Hidden_size, index, B, num_heads);
	//calucate_mhead_att_dec_2_kernel<<<blockPerGrid,64,0,gpu_info.s0>>>(d_alignments_dec, d_total_wx_t, Hidden_size, index, B, num_heads);
	CUDA_GET_LAST_ERROR("calucate_mhead_att_dec_kernel failed");
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
	for(int i=0; i<B; i++) {	
		cout<<"show d_normal_alignments[index] now "<<i<<endl;
		show_matrix(h_normal_alignments[B*index+i], num_heads, (index+1));
	}
*/	
	creat_total_c_t_dec_kernel<<<blockPerGrid,128,0,gpu_info.s0>>>(d_total_c_t, d_total_wx_t, d_normal_alignments, Hidden_size, index, B, num_heads);
	CUDA_GET_LAST_ERROR("creat_total_c_t_dec_kernel failed");
/*	
	cout<<"show H2H d_total_c_t[index] now "<<endl;
	show_matrix(h_total_c_t[index], Hidden_size, B);
*/	
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,Hidden_size,&alpha,d_output_transform_kernel,Hidden_size,h_total_c_t[index],Hidden_size,&beta,h_total_hdec_t[index],Hidden_size),"d_total_hdec_t error");
	

	///cout<<"show d_total_hdec_t[index] pre "<<endl;
	///show_matrix(h_total_hdec_t[index], Hidden_size, B);
	//cudaMemset(h_total_hdec_t[index],0,Hidden_size*minibatch_size*sizeof(float)); //?

	dim3 block_shape2(256,1,1);
	dim3 grid_shape2(B, (Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(h_total_hdec_t[index], d_output_transform_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
/*	
	cout<<"show H2H d_total_hdec_t[index] now "<<endl;
	show_matrix(h_total_hdec_t[index], Hidden_size, B);
*/	
	//cout<<"decoder Hidden_To_Hidden_Layer multi_head_att_dec cin"<<endl;	
	//cin>>a_test;	
}


void Hidden_To_Hidden_Layer::add_and_norm_dec(int index, int B) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_dec_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm_t, d_total_hdec_t, d_total_x_t, d_layer_norm_scale, d_layer_norm_bias, Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_dec_kernel failed");

	//cout<<"show H2H d_total_norm_t[index] now "<<endl;
	//show_matrix(h_total_norm_t[index], Hidden_size, B);
	
	//cin>>a_test;
		
}

void Hidden_To_Hidden_Layer::add_and_norm_1_dec(int index, int B) {
	
	cudaSetDevice(gpu_num);
	
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e2, 0);	
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_dec_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm1_t, d_total_h_t, d_total_norm_t, d_layer_norm_1_scale, d_layer_norm_1_bias, Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_dec_kernel failed");

	//cout<<"show H2H after encdec-att d_total_norm1_t[index] now "<<endl;
	//show_matrix(h_total_norm1_t[index], Hidden_size, B);
	
	//cin>>a_test;
		
}

void Hidden_To_Hidden_Layer::add_and_norm_2_dec(int index, int B) {
	
	cudaSetDevice(gpu_num);
	
	dim3 blockPerGrid(256,1,1);	
	residual_connection_and_norm_dec_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_norm2_t, d_total_feed_t, d_total_norm1_t, d_layer_norm_2_scale, d_layer_norm_2_bias, Hidden_size, index, B);
	CUDA_GET_LAST_ERROR("residual_connection_and_norm_dec_kernel failed");
/*
	cout<<"show H2H after encdec-att d_total_norm2_t[index] now "<<endl;
	show_matrix(h_total_norm2_t[index], Hidden_size, B);
*/	
	if(upper_layer.copy_h_t && upper_layer.hidden_layer) {

			cudaMemcpyAsync(upper_layer.hidden_layer->h_total_x_t[index], h_total_norm2_t[index], Hidden_size*B*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
		//upper_layer.hidden_layer->h_total_x_t[0] = h_total_h3_t[0];
		CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");
			
	}
	else if (upper_layer.copy_h_t && upper_layer.softmax && is_decoder) {
		
		cudaMemcpyAsync(upper_layer.softmax->d_single_x_t, h_total_norm2_t[index], Hidden_size*B*sizeof(float), cudaMemcpyDefault,gpu_info.s0);
		//upper_layer.hidden_layer->h_total_x_t[0] = h_total_h3_t[0];
		CUDA_GET_LAST_ERROR("cudaMemcpyAsync failed");

	}
	else {
		
	}
	
	
	//cin>>a_test;
		
}

void Hidden_To_Hidden_Layer::multi_head_att_encdec(int index, int B, int Sentence_len_src) {

	cudaSetDevice(gpu_num);
	float alpha = 1.0;
	float beta = 0.0;
	
	///cout<<"B: "<<B<<endl;
	///cout<<"index: "<<index<<endl;
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
	
	///cout<<"Sentence_len_src: "<<Sentence_len_src<<endl;	
	
	//cout<<"show decoder d_total_source_h_t[0] now "<<endl;
	//show_matrix(h_total_source_h_t[0], Hidden_size, 1);
	///cout<<"show decoder d_total_source_h_t[-1] now "<<endl;
	///show_matrix(h_total_source_h_t[Sentence_len_src-1], Hidden_size, 1);


	
	dim3 blockPerGrid(256,256);
	calucate_2hidden_matrix_encdec_kernel<<<blockPerGrid,256,0,gpu_info.s1>>>(d_total_wx_kvt,d_total_source_h_t,d_kv_kernel,d_kv_bias,Hidden_size,Sentence_len_src, B); //minibatch_size
	CUDA_GET_LAST_ERROR("calucate_2hidden_matrix_encdec_kernel failed");
	cudaEventRecord(gpu_info.e1, gpu_info.s1);
	
	//cout<<"show decoder d_total_wx_kvt[-1] now "<<endl;
	//show_matrix(h_total_wx_kvt[Sentence_len_src-1], Hidden_size, 1);

	
	//********************* decoder enc-dec attention *******************/
	cudaStreamWaitEvent(gpu_info.s2, gpu_info.e0, 0);
	cudaStreamWaitEvent(gpu_info.s2, gpu_info.e1, 0);
	dim3 blockPerGrid1(B,256,1);
	calucate_mhead_att_encdec_kernel<<<blockPerGrid1,64,0,gpu_info.s2>>>(d_alignments, d_single_wx_qt, d_total_wx_kvt, Hidden_size, Sentence_len_src, index, B, num_heads);
	
	///cout<<"show decoder encdec d_alignments[index] now "<<endl;
	///show_matrix(h_alignments[index], num_heads, Sentence_len_src);
	
	//cout<<"show enc-dec d_normal_alignments[index] pre "<<endl;
	//show_matrix(h_normal_alignments[index], num_heads, Sentence_len_src);
	
	dim3 grid_shape1(B,128,1);
	dim3 block_shape1(64,1,1);
	att_weight_norm_encdec_kernel<<<grid_shape1,block_shape1,0,gpu_info.s2>>>(d_normal_alignments, d_alignments, Hidden_size, Sentence_len_src, index, B, num_heads);
	CUDA_GET_LAST_ERROR("att_weight_norm_encdec_kernel failed");
	
	//cout<<"show enc-dec d_normal_alignments[index] now "<<endl;
	//show_matrix(h_normal_alignments[index], num_heads, Sentence_len_src);
	
	creat_total_c_t_encdec_kernel<<<blockPerGrid1,128,0,gpu_info.s2>>>(d_total_c_t, d_total_wx_kvt, d_normal_alignments, Hidden_size, Sentence_len_src, index, B, num_heads);
	CUDA_GET_LAST_ERROR("creat_total_c_t_encdec_kernel failed");
	
	///cout<<"show d_total_c_t[index] now "<<endl;
	///show_matrix(h_total_c_t[index], Hidden_size, B);
	
	cublasSetStream(gpu_info.handle, gpu_info.s2);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,Hidden_size,&alpha,d_output_transform_1_kernel,Hidden_size,h_total_c_t[index],Hidden_size,&beta,h_total_h_t[index],Hidden_size),"d_total_h_t error");
	
	//cudaMemset(h_total_h_t[index],0,Hidden_size*minibatch_size*sizeof(float));	

	//cout<<"show d_total_h_t[index] pre "<<endl;
	//show_matrix(h_total_h_t[index], Hidden_size, B);
	
	dim3 block_shape2(256,1,1);
	dim3 grid_shape2(B, (Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape2,block_shape2,0,gpu_info.s2>>>(h_total_h_t[index], d_output_transform_1_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");
	
	cudaEventRecord(gpu_info.e2, gpu_info.s2);
/*	
	cout<<"show H2H decoder dec-att and encdec att d_total_h_t[index] now "<<endl;
	show_matrix(h_total_h_t[index], Hidden_size, B);
*/
	//cin>>a_test;	

}

void Hidden_To_Hidden_Layer::feed_foward_dec(int index, int B) {

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
	dim3 grid_shape2(B, (4*Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_relu_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(h_total_relu_t[index],4*Hidden_size);
	CUDA_GET_LAST_ERROR("matrix_relu_kernel failed");


	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Hidden_size,B,4*Hidden_size,&alpha,d_ffn_2_kernel,4*Hidden_size,h_total_relu_t[index],4*Hidden_size,&beta,h_total_feed_t[index],Hidden_size),"d_total_relu_t error");

	//dim3 block_shape2(256,1,1);
	//dim3 grid_shape2(B, (Hidden_size + block_shape2.x -1)/block_shape2.x, 1); //here minibatch_size = 1;
	matrix_add_vector_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(h_total_feed_t[index], d_ffn_2_bias, Hidden_size);	
	CUDA_GET_LAST_ERROR("matrix_add_vector_kernel failed");

	///cout<<"show H2H after enc-dec att d_total_feed_t[index] now "<<endl;
	///show_matrix(h_total_feed_t[index], Hidden_size, B);
	
	//cin>>a_test;
		
	//calucate_matrix_multi_kernel<<<blockPerGrid,256,0,gpu_info.s0>>>(d_total_h_t, d_total_c_t, d_output_transform_kernel,d_output_transform_kernel,Hidden_size, Sentence_len, minibatch_size);
	

}


