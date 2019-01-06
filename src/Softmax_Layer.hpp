
//#include "Attention_Softmax_Layer.h"

void Softmax_Layer::init_Softmax_Layer(int Hidden_size, int minibatch_size, int vocab_size, int gpu_num) {

	//this->Embedding_size = Embedding_size;
	this->Hidden_size = Hidden_size;
	this->minibatch_size = minibatch_size;
	this->vocab_size = vocab_size;
	this->gpu_num = gpu_num;

	gpu_info.init(gpu_num);
	
	init_params();

}

void Softmax_Layer::load_weight(ifstream &input) {
	
	cudaSetDevice(gpu_num);

	// attention params
	read_matrix_GPU(d_output_bias,Embedding_size,1,input);
	read_matrix_GPU(d_W_c_p1,Embedding_size,LSTM_size,input);
	read_matrix_GPU(d_W_c_p2,Embedding_size,LSTM_size,input);
   	
	// softmax params
	read_matrix_GPU(d_D,vocab_size,Embedding_size,input);
	read_matrix_GPU(d_b_d,vocab_size,1,input);
	
	//cout<<"show d_D: "<<endl;
	//show_matrix(d_D, vocab_size, Embedding_size);
	
	

}

void Softmax_Layer::init_params() {
	
	cudaSetDevice(gpu_num);
	cudaMalloc((void**) &d_D, Hidden_size*vocab_size*sizeof(float));
	cudaMalloc((void**)&d_single_x_t, Hidden_size*minibatch_size*sizeof(float));

	//for small vocab
	cudaMalloc((void**)&d_D_small, 10000*Hidden_size*sizeof(float));
	cudaMalloc((void**)&d_tgt_vocab, 10000*sizeof(int));

	cudaMalloc((void**) &d_output_bias, Embedding_size*1*sizeof(float));
	cudaMalloc((void**) &d_W_c_p1, Embedding_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_c_p2, Embedding_size*LSTM_size*sizeof(float));

	cudaMalloc((void**)&d_final_temp_1, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_final_temp_2, LSTM_size*minibatch_size*sizeof(float));
	
	//cudaMalloc((void**) &d_D, vocab_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_b_d, vocab_size*1*sizeof(float));

	//node
	cudaMalloc((void**)&d_h_t_below, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_h_t_source, LSTM_size*200*sizeof(float)); // source maximum length 200

	cudaMalloc((void**)&d_alignment, minibatch_size*200*sizeof(float)); //
	cudaMalloc((void**)&d_normal_alignment, minibatch_size*200*sizeof(float)); //

	cudaMalloc((void**)&d_c_att_t, LSTM_size*minibatch_size*sizeof(float));
	

	//!!
	vector<float> ones(vocab_size,1);
	cudaMalloc((void**)&d_ones, vocab_size*sizeof(float));
	cudaMemcpy(d_ones, &ones[0], vocab_size*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_outputdist, vocab_size*minibatch_size*sizeof(float)); //
	cudaMalloc((void**)&d_outputdist_sum, minibatch_size*sizeof(float)); 
	cudaMalloc((void**)&d_logit_softmax, vocab_size*minibatch_size*sizeof(float));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Softmax_Layer::init_small_vocab(vector<int> tgt_vocab) {

	cudaSetDevice(gpu_num);
	
	//int VV = tgt_vocab.size();
	this->VV = tgt_vocab.size();
	//cout<<"VV in softmax: "<<VV<<endl;
	cudaMemcpy(d_tgt_vocab, &tgt_vocab[0], VV*sizeof(int), cudaMemcpyHostToDevice);
	dim3 block_shape0(256,1,1);
	dim3 grid_shape0(VV,(Hidden_size + block_shape0.x - 1)/block_shape0.x,1);

	lookup_kernel<<<grid_shape0,block_shape0,0,gpu_info.s0>>>(d_D_small,d_D,d_tgt_vocab,Hidden_size);
	
	//cout<<"d_D: "<<endl;
	//show_matrix(d_D, vocab_size, LSTM_size);	
	//cout<<"d_D_small: "<<endl;
	//show_matrix(d_D_small, VV, LSTM_size);	
	
/*
	dim3 block_shape1(1,1,1);
	dim3 grid_shape1(VV,1,1);
	lookup_rows<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_b_d_small,d_b_d,d_tgt_vocab,1,VV,vocab_size);
*/

	//cout<<"d_b_d: "<<endl;
	//show_matrix(d_b_d, vocab_size, 1);	
	//cout<<"d_b_d_small: "<<endl;
	//show_matrix(d_b_d_small, VV, 1);	
	
	CUDA_GET_LAST_ERROR("Attention_Softmax_Layer::init_small_vocab");
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	
}

void Softmax_Layer::softmax_forward_prop(int index, int B, bool is_small_vocab) {
	
	cudaSetDevice(gpu_num);
	
	//cout<<"show softmax_forward_prop d_D: "<<endl;
	//show_matrix(d_D, Hidden_size, vocab_size);
	
	//cout<<"show softmax_forward_prop d_single_x_t: "<<endl;
	//show_matrix(d_single_x_t, Hidden_size, B);
	
	float alpha = 1.0;
	float beta = 0.0;
	
	//cout<<"B: "<<B<<endl;
	//cout<<"index: "<<index<<endl;

	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	//CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,vocab_size,B,Hidden_size,&alpha,d_D,Hidden_size,d_single_x_t,Hidden_size,&beta,d_outputdist,vocab_size),"d_sing_h_t error");
	if (is_small_vocab) {
		CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,VV,B,Hidden_size,&alpha,d_D_small,Hidden_size,d_single_x_t,Hidden_size,&beta,d_outputdist,VV),"d_sing_h_t error");
	}
	else {
		this->VV = vocab_size;//!
		CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,vocab_size,B,Hidden_size,&alpha,d_D,Hidden_size,d_single_x_t,Hidden_size,&beta,d_outputdist,vocab_size),"d_sing_h_t error");
	}
	dim3 block_shape2(256,1,1);
	dim3 grid_shape2((VV+256-1)/256,B,1);

	exp_overflow_prevention<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(d_outputdist,VV); //exp(x-10)
	
	//cout<<"d_outputdist: "<<endl;
	//show_matrix(d_outputdist, vocab_size, B);
	
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,1,B,VV,&alpha,d_ones,1,d_outputdist,VV,&beta,d_outputdist_sum,1),"d_outputdist_sum error");
	
	divide<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(d_logit_softmax,d_outputdist,d_outputdist_sum,VV);
		
	CUDA_GET_LAST_ERROR("softmax_forward_prop failed");
	
	//cout<<"d_logit_softmax: "<<endl;
	//show_matrix(d_logit_softmax, vocab_size, B);
	
	//cin>>a_test;
}

