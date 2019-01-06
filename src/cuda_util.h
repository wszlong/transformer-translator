
#include <cublas_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void lookup_pa_kernel(float **d_lookup_total, float *d_W, int *d_wids, int Hidden_size, int Sentence_len, int minibatch_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    //int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	for(int j=blockIdx.x; j<minibatch_size*Sentence_len; j+=gridDim.x) {
		int minibatch_index = j % minibatch_size;
		int w_index = j / minibatch_size;
		//for(int j=threadIdx.x; j<)
		if(i<Hidden_size) {
		
			//d_lookup_total[w_index][IDX2C(i,minibatch_index,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)];
			d_lookup_total[w_index][IDX2C(i,minibatch_index,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)] * sqrt(float(Hidden_size)); //!!
		}
	}
}


__global__ void update_history_state_kernel(float **d_total_wx_tmp_t, float **d_total_wx_t, int *d_wids, int Hidden_size, int index, int minibatch_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    //int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	for(int j=blockIdx.x; j<minibatch_size*index; j+=gridDim.x) {
		int minibatch_index = j % minibatch_size;
		int w_index = j / minibatch_size;
		//for(int j=threadIdx.x; j<)
		if(i<Hidden_size) {

			d_total_wx_tmp_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] = d_total_wx_t[w_index][IDX2C(i,d_wids[minibatch_index],Hidden_size)]; //!!
			//d_total_wx_tmp_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] = d_total_wx_t[w_index][IDX2C(i,0,Hidden_size)]; //!!
		
		}
	}
}

__global__ void update_history_state_2_kernel(float *d_total_wx_tmp_t, float *d_total_wx_t, int *d_wids, int Hidden_size, int index, int minibatch_size)
{
    //int i = threadIdx.x + blockIdx.y*blockDim.x;
    //int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	//for(int j=blockIdx.x; j<minibatch_size*index; j+=gridDim.x) {
	int j=blockIdx.x;
		//int minibatch_index = j % minibatch_size;
		//int w_index = j / minibatch_size;
		//for(int j=threadIdx.x; j<)
		
	//if(i<Hidden_size) {
		for (int i=threadIdx.x; i<3*Hidden_size; i+=blockDim.x) {

			//d_total_wx_tmp_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] = d_total_wx_t[w_index][IDX2C(i,d_wids[j],Hidden_size)]; //!!
			//d_total_wx_tmp_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] = d_total_wx_t[w_index][IDX2C(i,0,Hidden_size)]; //!!
			d_total_wx_tmp_t[IDX2C(i,j,3*Hidden_size)] = d_total_wx_t[IDX2C(i,0,3*Hidden_size)];
		
		}
	
}


__global__ void target_space_and_pos_encoding_kernel(float **d_total_x_t, float *d_W, int target_space, int Hidden_size, int Sentence_len, int minibatch_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    //int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	int half_hidden_size = Hidden_size/2;
	int num_scale = Hidden_size/2 -1;
	for(int j=blockIdx.x; j<minibatch_size*Sentence_len; j+=gridDim.x) {
		int minibatch_index = j % minibatch_size;
		int w_index = j / minibatch_size;
		//for(int j=threadIdx.x; j<)
		if(i<half_hidden_size) {
			
			d_total_x_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] += (d_W[IDX2C(9,i,32)] + sin(w_index/(pow(10000,float(i)/float(num_scale))))); // multi times

		}
		else if(i<Hidden_size) {
			
			d_total_x_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] += (d_W[IDX2C(9,i,32)] + cos(w_index/(pow(10000,float((i-half_hidden_size))/float(num_scale))))); // 32 main dim
			
		}
	}
}

__global__ void position_encoding_kernel(float **d_total_x_t, int Hidden_size, int Sentence_len, int minibatch_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    //int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	int half_hidden_size = Hidden_size/2;
	int num_scale = Hidden_size/2 -1;
	for(int j=blockIdx.x; j<minibatch_size*Sentence_len; j+=gridDim.x) {
		int minibatch_index = j % minibatch_size;
		int w_index = j / minibatch_size;
		//for(int j=threadIdx.x; j<)
		if(i<half_hidden_size) {
			
			d_total_x_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] += sin(w_index/(pow(10000,float(i)/float(num_scale)))); // multi times

		}
		else if(i<Hidden_size) {
			
			d_total_x_t[w_index][IDX2C(i,minibatch_index,Hidden_size)] += cos(w_index/(pow(10000,float((i-half_hidden_size))/float(num_scale)))); // 32 main dim
			
		}
	}
}


__global__ void lookup_pa_kernel_2(float **d_lookup_total, float *d_W, int *d_wids, int Hidden_size, int Sentence_len, int minibatch_size)
{
    //int i = threadIdx.x + blockIdx.y*blockDim.x;
    //int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	for(int i=blockIdx.x; i<minibatch_size*Sentence_len; i+=gridDim.x) {
		int minibatch_index = i % minibatch_size;
		int w_index = i / minibatch_size;
		//for(int j=threadIdx.x; j<)
		for(int j=threadIdx.x; j<Hidden_size; j+=blockDim.x) {
		
			//d_lookup_total[w_index][IDX2C(j,minibatch_index,Hidden_size)] = d_W[IDX2C(j,d_wids[i],Hidden_size)];
			d_lookup_total[w_index][IDX2C(j,minibatch_index,Hidden_size)] = d_W[IDX2C(j,d_wids[IDX2C(minibatch_index,w_index,minibatch_size)],Hidden_size)];
		}
	}

}

__global__ void calucate_3hidden_matrix_kernel(float **d_total_3h_t, float *d_qkv, float **d_total_x_t, float *d_qkv_bias,int Hidden_size, int Sentence_len, int minibatch_size){
    __shared__ float buffer[256];
    const int tid = threadIdx.x;

    for(int h3_index=blockIdx.x; h3_index<3*Hidden_size;h3_index+=gridDim.x){
        for(int i=blockIdx.y;i<Sentence_len*minibatch_size;i+=gridDim.y){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;

            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
                buffer[tid] += d_qkv[IDX2C(j,h3_index,Hidden_size)]*d_total_x_t[s_index][IDX2C(j,m_index,Hidden_size)];
            }
            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float sum_k = buffer[0];   
            if(tid == 0){
                d_total_3h_t[s_index][IDX2C(h3_index,m_index,3*Hidden_size)] = sum_k+d_qkv_bias[h3_index];
            }
            __syncthreads();                 
        } 
    }
}


__global__ void calucate_2hidden_matrix_encdec_kernel(float **d_total_wx_kvt, float **d_total_source_h_t, float *d_kv, float *d_kv_bias,int Hidden_size, int Sentence_len_src, int minibatch_size){
    __shared__ float buffer[256];
    const int tid = threadIdx.x;

    for(int h_index=blockIdx.x; h_index<2*Hidden_size;h_index+=gridDim.x){
        for(int i=blockIdx.y;i<Sentence_len_src*minibatch_size;i+=gridDim.y){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;

            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
                //buffer[tid] += d_kv[IDX2C(j,h_index,Hidden_size)]*d_total_source_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
                buffer[tid] += d_kv[IDX2C(j,h_index,Hidden_size)]*d_total_source_h_t[s_index][IDX2C(j,0,Hidden_size)]; //minibatch_size=1 in encoder
            }
            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float sum_k = buffer[0];   
            if(tid == 0){
                d_total_wx_kvt[s_index][IDX2C(h_index,m_index,2*Hidden_size)] = sum_k+d_kv_bias[h_index];
            }
            __syncthreads();                 
        } 
    }
}




__global__ void calucate_matrix_multi_kernel(float **d_total_h_t, float **d_total_c_t, float *d_output_kernel, float *d_output_bias,int Hidden_size, int Sentence_len, int minibatch_size){
    __shared__ float buffer[256];
    const int tid = threadIdx.x;

    for(int h_index=blockIdx.x; h_index<Hidden_size;h_index+=gridDim.x){
        for(int i=blockIdx.y;i<Sentence_len*minibatch_size;i+=gridDim.y){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;

            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
                buffer[tid] += d_total_c_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_output_kernel[IDX2C(j,h_index,Hidden_size)];
            }
            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float sum_k = buffer[0];   
            if(tid == 0){
                d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = sum_k+d_output_bias[h_index];
            }
            __syncthreads();                 
        } 
    }
}


__global__ void calcuate_matrix_multi_and_relu_kernel(float **d_total_relu_t, float **d_total_h1_t, float *d_ffn_1_kernel, float *d_ffn_1_bias,int Hidden_size, int Sentence_len, int minibatch_size){
    __shared__ float buffer[256];
    const int tid = threadIdx.x;

    for(int h4_index=blockIdx.x; h4_index<4*Hidden_size;h4_index+=gridDim.x){
        for(int i=blockIdx.y;i<Sentence_len*minibatch_size;i+=gridDim.y){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;

            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
                buffer[tid] += d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_ffn_1_kernel[IDX2C(j,h4_index,Hidden_size)];
            }
            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float sum_k = buffer[0];   
            if(tid == 0){
                d_total_relu_t[s_index][IDX2C(h4_index,m_index,4*Hidden_size)] = sum_k + d_ffn_1_bias[h4_index];
				//relu
				if (d_total_relu_t[s_index][IDX2C(h4_index,m_index,4*Hidden_size)] < 0) {
					d_total_relu_t[s_index][IDX2C(h4_index,m_index,4*Hidden_size)] = 0;
				}
            }
            __syncthreads(); 

			//matrix multi second!!
			/*
            buffer[tid] = 0;
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
                buffer[tid] += d_total_relu_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_ffn_2_kernel[IDX2C(j,h4_index,Hidden_size)];
            }
            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float sum_k = buffer[0];   
			*/

        } 
    }
}

__global__ void calcuate_matrix_multi_after_relu_kernel(float **d_total_h2_t, float **d_total_relu_t, float *d_ffn_2_kernel, float *d_ffn_2_bias,int Hidden_size, int Sentence_len, int minibatch_size){
    __shared__ float buffer[256];
    const int tid = threadIdx.x;

    for(int h_index=blockIdx.x; h_index<Hidden_size;h_index+=gridDim.x){
        for(int i=blockIdx.y;i<Sentence_len*minibatch_size;i+=gridDim.y){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;

            for (int j=threadIdx.x;j<4*Hidden_size;j+=blockDim.x){
                buffer[tid] += d_total_relu_t[s_index][IDX2C(j,m_index,4*Hidden_size)] * d_ffn_2_kernel[IDX2C(j,h_index,4*Hidden_size)];
            }
            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float sum_k = buffer[0];   
            if(tid == 0){
                d_total_h2_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = sum_k+d_ffn_2_bias[h_index];
            }
            __syncthreads(); 
			
        } 
    }
}



__global__ void multi_head_attention_kernel(float **d_total_h_t_2, float **d_total_h_t_1, int Hidden_size, int Sentence_len, int minibatch_size, int num_heads) 
{
	
    //int i = threadIdx.x + blockIdx.y*blockDim.x;
	
	//__shared__ float buffer[256][256];	
	__shared__ float buffer[256];	
	const int tid = threadIdx.x;

	for(int i=blockIdx.x; i<minibatch_size*Sentence_len; i+=gridDim.x) {
		int minibatch_index = i % minibatch_size;
		int w_index = i / minibatch_size;
		
		for(int j=threadIdx.x; j<Hidden_size; j+=blockDim.x) {

			buffer[tid] += d_total_h_t_1[0][IDX2C(j,minibatch_index,3*Hidden_size)] * d_total_h_t_1[w_index][IDX2C(j+Hidden_size,minibatch_index,3*Hidden_size)];
			
			//buffer[blockIdx.y][tid] += d_total_h_t_1[blockIdx.y][IDX2C(j,minibatch_index,3*Hidden_size)] * d_total_h_t_1[w_index][IDX2C(j+Hidden_size,minibatch_index,3*Hidden_size)];
			//buffer[blockIdx.y][tid] += d_total_h_t_1[blockIdx.y][IDX2C(j,minibatch_index,2*Hidden_size)] * d_total_h_t_1[w_index][IDX2C(j+Hidden_size,minibatch_index,2*Hidden_size)];
		}
	}

}

__global__ void multi_head_attention_kernel_2(float **d_results, float **d_total_h_t_2, float **d_total_wx_t, int Hidden_size, int Sentence_len, int minibatch_size, int num_heads) 
{
	
    //int i = threadIdx.x + blockIdx.y*blockDim.x;
	
	__shared__ float buffer[128];	
	//__shared__ float buffer[256];	
	
	const int tid = threadIdx.x;
	int w_index=blockIdx.x;
	//const int tidy = threadIdx.y;

	for(int i=blockIdx.y; i<minibatch_size*Sentence_len; i+=gridDim.y) {
		int minibatch_index = i % minibatch_size;
		int w_index_2 = i / minibatch_size;
		buffer[tid]= 0;

		for(int j=threadIdx.x; j<Hidden_size; j+=blockDim.x) {
//			for(int k=threadIdx.y; k<Hidden_size; k+=blockDim.y) {

			//int hidden_index = j % Hidden_size;
			//int w_index_2 = j / Hidden_size;
			
			// x y
			buffer[tid] += d_total_wx_t[w_index][IDX2C(j,minibatch_index,3*Hidden_size)] * d_total_wx_t[w_index_2][IDX2C(j+Hidden_size,minibatch_index,3*Hidden_size)];
		}
	   
	    __syncthreads();
	    for(int stride = 128/2;stride > 0;stride>>=1){
		   if(tid < stride){
			   buffer[tid] += buffer[stride + tid];
		   }
		   __syncthreads();
	    }
	    __syncthreads();
														
		float sum_k = buffer[0];   
		if(tid == 0) {
			d_results[w_index][IDX2C(minibatch_index,w_index_2,minibatch_size)] = sum_k;
		}
		__syncthreads();
	}

}

__global__ void multi_head_attention_kernel_3(float **d_results, float **d_total_h_t_2, float **d_total_wx_t, int Hidden_size, int Sentence_len, int minibatch_size, int num_heads) 
{
	
    //int i = threadIdx.x + blockIdx.y*blockDim.x;
	
	//__shared__ float buffer[num_heads][128];	
	__shared__ float buffer[8][64];	
	//__shared__ float buffer[256];	
	
	int Hidden_size_single = Hidden_size / num_heads;
	
	const int tid = threadIdx.x;
	int w_index=blockIdx.x;
	//const int tidy = threadIdx.y;

	for(int i=blockIdx.y; i<minibatch_size*Sentence_len; i+=gridDim.y) {
		int minibatch_index = i % minibatch_size;
		int w_index_2 = i / minibatch_size;
		for(int k=0; k<num_heads; k++) {
			buffer[k][tid]= 0;
		}

		for(int j=threadIdx.x; j<Hidden_size; j+=blockDim.x) {
		//for(int k=threadIdx.y; k<Hidden_size_single; k+=blockDim.y) {

			//int hidden_index = j % Hidden_size;
			//int w_index_2 = j / Hidden_size;
			
			//Hidden_size_single_index = j % Hidden_size_single;
			//num_head_index = j / Hidden_size_single;
			
			//buffer[k][tid] += d_total_wx_t[w_index][IDX2C(j,minibatch_index,3*Hidden_size)] * d_total_wx_t[w_index_2][IDX2C(j+Hidden_size,minibatch_index,3*Hidden_size)];
			
			for(int k=0; k<num_heads; k++) {
				if(j>k*Hidden_size_single && j<(k+1)*Hidden_size_single) {
					buffer[k][tid] += d_total_wx_t[w_index][IDX2C(j,minibatch_index,3*Hidden_size)] * d_total_wx_t[w_index_2][IDX2C(j+Hidden_size,minibatch_index,3*Hidden_size)];
				}
			}

			//if (j<Hidden_size_single) {
			//	buffer[0][tid] += d_total_h_t_1[w_index][IDX2C(j,minibatch_index,2*Hidden_size)] * d_total_h_t_1[w_index_2][IDX2C(j+Hidden_size,minibatch_index,2*Hidden_size)];
			//}
			//else if {j<2*Hidden_size_single} {		}
		}
	   
	    __syncthreads();
	    for(int stride = 64/2;stride > 0;stride>>=1){
			if(tid < stride){
				for(int k=0; k<num_heads; k++) {
			   		buffer[k][tid] += buffer[k][stride + tid];
		   		__syncthreads(); // add by me
				}
		   	}
		   	__syncthreads();
	    }
	    __syncthreads();
		
		for(int k=0; k<num_heads; k++) {		
			float sum_k = buffer[k][0];   
			if(tid == 0) {
				d_results[w_index][IDX2C(minibatch_index+k,w_index_2,minibatch_size*num_heads)] = sum_k;
			}
			__syncthreads();
		}
	}

}


__global__ void multi_head_attention_kernel_4(float **d_results, float **d_total_h_t_2, float **d_total_wx_t, int Hidden_size, int Sentence_len, int minibatch_size, int num_heads) 
{
	
    //int i = threadIdx.x + blockIdx.y*blockDim.x;
	
	//__shared__ float buffer[num_heads][128];	
	__shared__ float buffer[8][64];	
	//__shared__ float buffer[256];	
	
	int Hidden_size_single = Hidden_size / num_heads;
	
	const int tid = threadIdx.y;
	int w_index=blockIdx.x;
	
	const int num_heads_index = threadIdx.x;
	
	//int num_heads_index = threadIdx.x;
	//const int tidy = threadIdx.y;

	for(int i=blockIdx.y; i<minibatch_size*Sentence_len; i+=gridDim.y) {
		//for(int j=blockIdx.x; j<num_heads*Sentence_len; j+=gridDim.x) {

		int minibatch_index = i % minibatch_size;
		int w_index_2 = i / minibatch_size;

		//}
		buffer[num_heads_index][tid]= 0;

		//for(int j=threadIdx.x; j<num_heads; j+=blockDim.x) {
		for(int k=threadIdx.y; k<Hidden_size; k+=blockDim.y) {

			//int hidden_index = j % Hidden_size;
			//int num_heads_index = k / Hidden_size_single;
			
			//Hidden_size_single_index = j % Hidden_size_single;
			//num_heads_index = j / Hidden_size_single;
			if (num_heads_index == k / Hidden_size_single) {

				buffer[num_heads_index][tid] += d_total_wx_t[w_index][IDX2C(k,minibatch_index,3*Hidden_size)] * d_total_wx_t[w_index_2][IDX2C(k+Hidden_size,minibatch_index,3*Hidden_size)];
			}
		}
	   
	    __syncthreads();
	    for(int stride = 64/2;stride > 0;stride>>=1){
			if(tid < stride){
				//}
				buffer[num_heads_index][tid] += buffer[num_heads_index][stride + tid];
		   	}
		   	__syncthreads();
	    }
	    __syncthreads();
		
		//float sum_k = buffer[num_heads_index][0];   
		
		if(tid == 0) {
			d_results[w_index][IDX2C(IDX2C(minibatch_index,num_heads_index,minibatch_size),w_index_2,minibatch_size*num_heads)] = buffer[num_heads_index][0];
		}
		__syncthreads();
		
	}

}


__global__ void calucate_mhead_att_kernel(float **d_mhead_att,float **d_total_3h_t,int Hidden_size, int Sentence_len, int minibatch_size,int mhead_num){
    __shared__ float buffer[64];
    const int tid = threadIdx.x;
	int Hidden_size_perHead = Hidden_size/mhead_num;

    for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index=i%minibatch_size;
         int s_index_1=i/minibatch_size;
         for(int j=blockIdx.y;j<mhead_num*Sentence_len;j+=gridDim.y){
             int mhead_index=j%mhead_num;
             int s_index_2=j/mhead_num;
             
             buffer[tid] = 0;
             
             for (int k =threadIdx.x; k<Hidden_size_perHead; k+=blockDim.x){  
                 int q_index=k+mhead_index*Hidden_size_perHead;  
                 int k_index=k+mhead_index*Hidden_size_perHead+Hidden_size;
                 buffer[tid] += d_total_3h_t[s_index_1][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_3h_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
             }
             __syncthreads();
            for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
            }
            __syncthreads();

            float sum_k = buffer[0];
            if(tid == 0){
                d_mhead_att[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,mhead_num)] = sum_k/sqrt(float(Hidden_size_perHead));   // !!
                //d_mhead_att[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,mhead_num)] = 2;   // !!
            }
            __syncthreads();
         }
    } 
}

//!!!!!!!!!!!!!!!!!!!!!
//d_alignments, d_total_wx_t, Hidden_size, index, B, num_heads
__global__ void calucate_mhead_att_dec_kernel(float **d_mhead_att,float **d_total_wx_t,int Hidden_size, int index, int minibatch_size, int num_heads){
    __shared__ float buffer[64];
    const int tid = threadIdx.x;
	int Hidden_size_perHead = Hidden_size/num_heads;

    for(int i=blockIdx.x; i<minibatch_size; i+=gridDim.x){
         //int m_index = i;
		 int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
         for(int j=blockIdx.y;j<num_heads*(index+1);j+=gridDim.y){
             int mhead_index=j%num_heads;
             int s_index_2=j/num_heads;
             
             buffer[tid] = 0;
             
             for (int k =threadIdx.x; k<Hidden_size_perHead; k+=blockDim.x){  
                 int q_index=k+mhead_index*Hidden_size_perHead;
                 int k_index=k+mhead_index*Hidden_size_perHead+Hidden_size;
                 //buffer[tid] += d_total_wx_t[index][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_wx_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
                 buffer[tid] += d_total_wx_t[index][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_wx_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
             }
             __syncthreads();
            for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
            }
            __syncthreads();

            float sum_k = buffer[0];
            if(tid == 0){
                d_mhead_att[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = sum_k/sqrt(float(Hidden_size_perHead));
                //d_mhead_att[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = 2.0;
                //d_mhead_att[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = 2.0;
            }
            __syncthreads();
         }
    } 
}


//d_alignments, d_single_wx_qt, d_total_wx_kvt, Hidden_size, Sentence_len_src, B, num_heads
__global__ void calucate_mhead_att_encdec_kernel(float **d_mhead_att,float *d_single_wx_qt,float **d_total_wx_kvt, int Hidden_size, int Sentence_len_src, int index, int minibatch_size, int num_heads){
    __shared__ float buffer[64];
    const int tid = threadIdx.x;
	int Hidden_size_perHead = Hidden_size/num_heads;

    //for(int i=blockIdx.x; i<minibatch_size*(index+1); i+=gridDim.x){
         int m_index = blockIdx.x;
		 //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
         for(int j=blockIdx.y;j<num_heads*Sentence_len_src;j+=gridDim.y){
             int mhead_index=j%num_heads;
             int s_index_2=j/num_heads;
             
             buffer[tid] = 0;
             
             for (int k =threadIdx.x; k<Hidden_size_perHead; k+=blockDim.x){  
                 int q_index=k+mhead_index*Hidden_size_perHead;
                 int k_index=k+mhead_index*Hidden_size_perHead; //!!
                 buffer[tid] += d_single_wx_qt[IDX2C(q_index,m_index,Hidden_size)]*d_total_wx_kvt[s_index_2][IDX2C(k_index,m_index,2*Hidden_size)];
             }
             __syncthreads();
            for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
            }
            __syncthreads();

            float sum_k = buffer[0];
            if(tid == 0){
                d_mhead_att[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = sum_k/sqrt(float(Hidden_size_perHead));
            }
            __syncthreads();
         }
    //} 
}

//
__global__ void att_weight_normalization_kernel(float **d_normal_alignment,float **d_alignment,int Hidden_size, int Sentence_len, int minibatch_size,int num_heads){
    __shared__ float buffer[64];
    const int tid = threadIdx.x;
	//int Hidden_size_perHead = Hidden_size/num_heads;

    for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index=i%minibatch_size;
         int s_index_1=i/minibatch_size;
         for(int j=blockIdx.y;j<num_heads;j+=gridDim.y){
             int mhead_index=j;
             //int s_index_2=j/num_heads;
             
             buffer[tid] = -FLT_MAX;

			 for (int k =threadIdx.x; k<Sentence_len; k+=blockDim.x){
				 float z = d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
				 if(buffer[tid] < z) {
				 	 buffer[tid] = z; //get maximum
				 }
			 } 
             __syncthreads();
			 for(int stride=64/2; stride>0; stride>>=1) {
			 	 if (tid<stride) {
				 	buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid]; //!!
				 }
			 	 __syncthreads();
			 }
			 __syncthreads();
			 
			 float max_k = buffer[0];
			 __syncthreads();


             buffer[tid] = 0;

             for (int k =threadIdx.x; k<Sentence_len; k+=blockDim.x){  
                 //int q_index=threadIdx.x+mhead_index*Hidden_size_perHead;
                 //int k_index=threadIdx.x+mhead_index*Hidden_size_perHead+Hidden_size;
				 d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = exp(d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)]-max_k); 
				 buffer[tid] += d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
                 //buffer[tid] += d_total_3h_t[s_index_1][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_3h_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
             }
             __syncthreads();
            for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
            }
            __syncthreads();

            float sum_k = buffer[0];
            //if(tid == 0){  //!!!
            	for (int k =threadIdx.x; k<Sentence_len; k+=blockDim.x){  
					d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] / sum_k;
                //d_mhead_att[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = sum_k;
				}
			//}
            __syncthreads();
         }
    } 
}

__global__ void att_weight_norm_dec_kernel(float **d_normal_alignment,float **d_alignment,int Hidden_size, int index, int minibatch_size,int num_heads){
    __shared__ float buffer[64];
    const int tid = threadIdx.x;
	//int Hidden_size_perHead = Hidden_size/num_heads;

    for(int i=blockIdx.x; i<minibatch_size;i+=gridDim.x){
         int m_index = i;
		 //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
         for(int j=blockIdx.y;j<num_heads;j+=gridDim.y){
             int mhead_index=j;
             //int s_index_2=j/num_heads;
             
			 buffer[tid] = -FLT_MAX;

			 for (int k =threadIdx.x; k<(index+1); k+=blockDim.x){
				 float z = d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
				 if(buffer[tid] < z) {
				 	 buffer[tid] = z; //get maximum
				 }
			 } 
             __syncthreads();
			 for(int stride=64/2; stride>0; stride>>=1) {
			 	 if (tid<stride) {
				 	buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid]; //!!
				 }
			 	 __syncthreads();
			 }
			 __syncthreads();
			 
			 float max_k = buffer[0];
			 __syncthreads();
             
             
			 buffer[tid] = 0;
             
             for (int k =threadIdx.x; k<(index+1); k+=blockDim.x){  
				 d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = exp(d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)]-max_k); 
				 buffer[tid] += d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
                 //buffer[tid] += d_total_3h_t[s_index_1][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_3h_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
             }
             __syncthreads();
            for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
            }
            __syncthreads();

            float sum_k = buffer[0];
            //if(tid == 0){  //!!!!
            	for (int k =threadIdx.x; k<(index+1); k+=blockDim.x){  
					d_normal_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)] / sum_k;
                //d_mhead_att[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = sum_k;
				}
			//}
            __syncthreads();
         }
    } 
}


__global__ void att_weight_norm_encdec_kernel(float **d_normal_alignment,float **d_alignment,int Hidden_size, int Sentence_len_src, int index, int minibatch_size,int num_heads){
    __shared__ float buffer[64];
    const int tid = threadIdx.x;
	//int Hidden_size_perHead = Hidden_size/num_heads;

    //for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index = blockIdx.x;
		 //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
         for(int j=blockIdx.y;j<num_heads;j+=gridDim.y){
             int mhead_index=j;
             //int s_index_2=j/num_heads;
             
			 buffer[tid] = -FLT_MAX;

			 for (int k =threadIdx.x; k<Sentence_len_src; k+=blockDim.x){
				 float z = d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
				 if(buffer[tid] < z) {
				 	 buffer[tid] = z; //get maximum
				 }
			 } 
             __syncthreads();
			 for(int stride=64/2; stride>0; stride>>=1) {
			 	 if (tid<stride) {
				 	buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid]; //!!
				 }
			 	 __syncthreads();
			 }
			 __syncthreads();
			 
			 float max_k = buffer[0];
			 __syncthreads();
             

             buffer[tid] = 0;
             
             for (int k =threadIdx.x; k<Sentence_len_src; k+=blockDim.x){  
				 d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = exp(d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)]-max_k); 
				 buffer[tid] += d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
                 //buffer[tid] += d_total_3h_t[s_index_1][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_3h_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
             }
             __syncthreads();
            for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
            }
            __syncthreads();

            float sum_k = buffer[0];
            //if(tid == 0){  //!!!!
            	for (int k =threadIdx.x; k<Sentence_len_src; k+=blockDim.x){  
					d_normal_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = d_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,k,num_heads)] / sum_k;
                //d_mhead_att[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] = sum_k;
				}
			//}
            __syncthreads();
         }
    //} 
}

__global__ void creat_total_c_t_kernel(float **d_total_c_t,float **d_total_wx_t,float **d_normal_alignment,int Hidden_size, int Sentence_len, int minibatch_size,int num_heads){
    
	__shared__ float buffer[64];
    const int tid = threadIdx.x;
	
	int Hidden_size_perHead = Hidden_size/num_heads;
	//float buffer[num_heads][Hidden_size_perHead];

    for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index=i%minibatch_size;
         int s_index_1=i/minibatch_size;
		 //!
		 //d_total_c_t[s_index_1] = 0;
         
		 for(int j=blockIdx.y;j<num_heads*Hidden_size_perHead;j+=gridDim.y){
             //int mhead_index=j%num_heads; //error!!!!
             int mhead_index=j/Hidden_size_perHead;
             //int h_index=j/num_heads;
      		 
	  		 //zero!!
			 //if(tid ==0) {
			 //	d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = 0;
			 //}
             buffer[tid] = 0;
			 
			 int v_index=j+2*Hidden_size;
             
             for (int k =threadIdx.x; k<Sentence_len; k+=blockDim.x){  
				
				 buffer[tid] += d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] * d_total_wx_t[k][IDX2C(v_index,m_index,3*Hidden_size)];
				 //d_total_c_t[s_index_1][mhead_index*Hidden_size_perHead+k,m_index,Hidden_size] = 
             }
             
			 __syncthreads();
             for(int stride = 64/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
             }
             __syncthreads();

             float sum_k = buffer[0];
             if(tid == 0){  //!!!!
				 d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = sum_k;
			 }
             __syncthreads();
			
         }
    } 
}

__global__ void creat_total_c_t_dec_kernel(float **d_total_c_t,float **d_total_wx_t,float **d_normal_alignment,int Hidden_size, int s_index_1, int minibatch_size,int num_heads){
    
	__shared__ float buffer[128];
    const int tid = threadIdx.x;
	
	int Hidden_size_perHead = Hidden_size/num_heads;
	//float buffer[num_heads][Hidden_size_perHead];

    for(int i=blockIdx.x; i<minibatch_size;i+=gridDim.x){
         //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
		 int m_index = i;
         
		 for(int j=blockIdx.y;j<num_heads*Hidden_size_perHead;j+=gridDim.y){
             //int mhead_index=j%num_heads; //error
             int mhead_index=j/Hidden_size_perHead;
      		 
	  		 //zero!!
			 //d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = 0;
             buffer[tid] = 0;
			 
			 int v_index=j+2*Hidden_size;
             
             for (int k =threadIdx.x; k<(s_index_1+1); k+=blockDim.x){  
				 //buffer[tid] += d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] * d_total_wx_t[k][IDX2C(v_index,m_index,3*Hidden_size)];
				 buffer[tid] += d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] * d_total_wx_t[k][IDX2C(v_index,m_index,3*Hidden_size)];
             }
             
			 __syncthreads();
             for(int stride = 128/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
             }
             __syncthreads();

             float sum_k = buffer[0];
             if(tid == 0){  //!!!!
				 d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = sum_k;
				 //d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = 0.3;
			 }
             __syncthreads();
				 
         }
    } 
}

__global__ void creat_total_c_t_encdec_kernel(float **d_total_c_t,float **d_total_wx_kvt,float **d_normal_alignment,int Hidden_size, int Sentence_len_src, int s_index_1, int minibatch_size,int num_heads){
    
	__shared__ float buffer[128];
    const int tid = threadIdx.x;
	
	int Hidden_size_perHead = Hidden_size/num_heads;
	//float buffer[num_heads][Hidden_size_perHead];

    for(int i=blockIdx.x; i<minibatch_size;i+=gridDim.x){
         //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
		 int m_index = i;
         
		 for(int j=blockIdx.y;j<num_heads*Hidden_size_perHead;j+=gridDim.y){
             //int mhead_index=j%num_heads;  //error
             int mhead_index=j/Hidden_size_perHead;
      		 
	  		 //zero!!
			 //d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = 0;
             buffer[tid] = 0;
			 
			 int v_index=j+Hidden_size;
             
             //for (int k =threadIdx.x; k<(s_index_1+1); k+=blockDim.x){ //error 
             for (int k =threadIdx.x; k<Sentence_len_src; k+=blockDim.x){  
				 buffer[tid] += d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] * d_total_wx_kvt[k][IDX2C(v_index,m_index,2*Hidden_size)];
             }
             
			 __syncthreads();
             for(int stride = 128/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }
                __syncthreads();
             }
             __syncthreads();

             float sum_k = buffer[0];
             if(tid == 0){  //!!!!
				 d_total_c_t[s_index_1][IDX2C(j,m_index,Hidden_size)] = sum_k;
			 }
             __syncthreads();
				 
         }
    } 
}

//error
__global__ void creat_total_c_t_error_kernel(float **d_total_c_t,float **d_total_wx_t,float **d_normal_alignment,int Hidden_size, int Sentence_len, int minibatch_size,int num_heads){
    
	//__shared__ float buffer[num_heads][64];
    //const int tid = threadIdx.x;
	int Hidden_size_perHead = Hidden_size/num_heads;
	//float buffer[num_heads][Hidden_size_perHead];

    for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index=i%minibatch_size;
         int s_index_1=i/minibatch_size;
		 //!
		 //d_total_c_t[s_index_1] = 0;
         
		 for(int j=blockIdx.y;j<num_heads*Sentence_len;j+=gridDim.y){
             int mhead_index=j%num_heads;
             int s_index_2=j/num_heads;
      		 
	  		 //zero!!!
             
			 //for (int k =threadIdx.y; k<Hidden_size; k+=blockDim.y){  
			 //	d_total_c_t[s_index_1][IDX2C(k,m_index,Hidden_size)] = 0;
			 //}
             //__syncthreads();

             //buffer[mhead_index][tid] = 0;
             
             for (int k =threadIdx.x; k<Hidden_size_perHead; k+=blockDim.x){  
                 int v_index=k+mhead_index*Hidden_size_perHead+2*Hidden_size;
                 //int k_index=threadIdx.x+mhead_index*Hidden_size_perHead+Hidden_size;
				 
				 //d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)] = exp(d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)]); 
				 //buffer[tid] += d_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,k,num_heads)];
                 //buffer[tid] += d_total_3h_t[s_index_1][IDX2C(q_index,m_index,3*Hidden_size)]*d_total_3h_t[s_index_2][IDX2C(k_index,m_index,3*Hidden_size)];
				
				 //buffer[k] += d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] * d_total_wx_t[s_index_1][IDX2C(v_index,m_index,3*Hidden_size)]
				 
				 //d_total_c_t[s_index_1][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] = 0;
				 d_total_c_t[s_index_1][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] += d_normal_alignment[IDX2C(m_index,s_index_1,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] * d_total_wx_t[s_index_2][IDX2C(v_index,m_index,3*Hidden_size)];

				 //d_total_c_t[s_index_1][mhead_index*Hidden_size_perHead+k,m_index,Hidden_size] = 
             }
			
			 //d_total_c_t[s_index_1][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] = buffer[]
				 
         }
    } 
}

__global__ void creat_total_c_t_dec_error_kernel(float **d_total_c_t,float **d_total_wx_t,float **d_normal_alignment,int Hidden_size, int index, int minibatch_size,int num_heads){
    
	//__shared__ float buffer[num_heads][64];
    //const int tid = threadIdx.x;
	int Hidden_size_perHead = Hidden_size/num_heads;
	//float buffer[num_heads][Hidden_size_perHead];

    //for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index = blockIdx.x;
		 //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
         
		 for(int j=blockIdx.y;j<num_heads*(index+1);j+=gridDim.y){
             int mhead_index=j%num_heads;
             int s_index_2=j/num_heads;
             //buffer[mhead_index][tid] = 0;
             
             for (int k =threadIdx.x; k<Hidden_size_perHead; k+=blockDim.x){  
                 int v_index=k+mhead_index*Hidden_size_perHead+2*Hidden_size;
                 //int k_index=threadIdx.x+mhead_index*Hidden_size_perHead+Hidden_size;
				 
				 d_total_c_t[index][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] = 0;
				 d_total_c_t[index][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] += d_normal_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] * d_total_wx_t[s_index_2][IDX2C(v_index,m_index,3*Hidden_size)];

				 //d_total_c_t[s_index_1][mhead_index*Hidden_size_perHead+k,m_index,Hidden_size] = 
             }
			
			 //d_total_c_t[s_index_1][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] = buffer[]
				 
         }
    //} 
}

__global__ void creat_total_c_t_encdec_error_kernel(float **d_total_c_t,float **d_total_wx_kvt,float **d_normal_alignment,int Hidden_size, int Sentence_len_src, int index, int minibatch_size,int num_heads){
    
	//__shared__ float buffer[num_heads][64];
    //const int tid = threadIdx.x;
	int Hidden_size_perHead = Hidden_size/num_heads;
	//float buffer[num_heads][Hidden_size_perHead];

    //for(int i=blockIdx.x; i<minibatch_size*Sentence_len;i+=gridDim.x){
         int m_index = blockIdx.x;
		 //int m_index=i%minibatch_size;
         //int s_index_1=i/minibatch_size;
         
		 for(int j=blockIdx.y;j<num_heads*Sentence_len_src;j+=gridDim.y){
             int mhead_index=j%num_heads;
             int s_index_2=j/num_heads;
             //buffer[mhead_index][tid] = 0;
             
             for (int k =threadIdx.x; k<Hidden_size_perHead; k+=blockDim.x){  
                 int v_index=k+mhead_index*Hidden_size_perHead+Hidden_size;
                 //int k_index=threadIdx.x+mhead_index*Hidden_size_perHead+Hidden_size;
				 
				 d_total_c_t[index][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] = 0;
				 d_total_c_t[index][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] += d_normal_alignment[IDX2C(m_index,index,minibatch_size)][IDX2C(mhead_index,s_index_2,num_heads)] * d_total_wx_kvt[s_index_2][IDX2C(v_index,m_index,2*Hidden_size)];

				 //d_total_c_t[s_index_1][mhead_index*Hidden_size_perHead+k,m_index,Hidden_size] = 
             }
			
			 //d_total_c_t[s_index_1][IDX2C(mhead_index*Hidden_size_perHead+k,m_index,Hidden_size)] = buffer[]
				 
         }
    //} 

}

//problem?? NAN
__global__ void residual_connection_and_norm_pre_kernel(float **d_total_h1_t, float **d_total_h_t, float **d_total_x_t, float *d_layer_norm_scale, float *d_layer_norm_bias,int Hidden_size, int Sentence_len, int minibatch_size){
    __shared__ float buffer[256];
    //__shared__ float buffer1[256];
    const int tid = threadIdx.x;

    for(int h_index=blockIdx.x; h_index<Hidden_size;h_index+=gridDim.x){
        for(int i=blockIdx.y;i<Sentence_len*minibatch_size;i+=gridDim.y){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;
			
			//if (tid==0)	{
			//	d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] + d_total_x_t[s_index][IDX2C(h_index,m_index,Hidden_size)];
			//}
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] +=  d_total_x_t[s_index][IDX2C(j,m_index,Hidden_size)];
				buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
				//buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] + d_total_x_t[s_index][IDX2C(j,m_index,Hidden_size)]; //error !!!!!!!!!!!!
			}

            //for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
            //    buffer[tid] += d_total_c_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_output_kernel[IDX2C(j,h_index,Hidden_size)];
            //}

            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float ave_k = buffer[0] / Hidden_size;
			//minus ave	
			if (tid == 0) {
				d_total_h1_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] - ave_k;
			}
            __syncthreads();                 
			
			
			//pow and sum
            buffer[tid] = 0;
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] = pow(d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)],2);
				buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
			}

			__syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float std_k = sqrt(buffer[0] / Hidden_size);
			// (x-ave)/std
			if (tid == 0) {
            	d_total_h1_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h1_t[s_index][IDX2C(h_index,m_index,Hidden_size)] / std_k;
			}			
            __syncthreads();                 
			
			// x*d_layer_norm_scale + d_layer_norm_bias
			if (tid==0) {
				d_total_h1_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h1_t[s_index][IDX2C(h_index,m_index,Hidden_size)] * d_layer_norm_scale[h_index] + d_layer_norm_bias[h_index];
			}
            __syncthreads(); //!          
			
        } 
    }
}

__global__ void residual_connection_and_norm_kernel(float **d_total_h1_t, float **d_total_h_t, float **d_total_x_t, float *d_layer_norm_scale, float *d_layer_norm_bias,int Hidden_size, int Sentence_len, int minibatch_size){
    __shared__ float buffer[256];
    //__shared__ float buffer1[256];
    const int tid = threadIdx.x;

    //for(int h_index=blockIdx.x; h_index<Hidden_size;h_index+=gridDim.x){
        for(int i=blockIdx.x;i<Sentence_len*minibatch_size;i+=gridDim.x){
            int s_index=i/minibatch_size;
            int m_index=i%minibatch_size;
            buffer[tid] = 0;
			
			//if (tid==0)	{
			//	d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] + d_total_x_t[s_index][IDX2C(h_index,m_index,Hidden_size)];
			//}
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] +=  d_total_x_t[s_index][IDX2C(j,m_index,Hidden_size)];  //!!
				buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
				//buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] + d_total_x_t[s_index][IDX2C(j,m_index,Hidden_size)]; //error !!!!!!!!!!!!
			}

            //for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
            //    buffer[tid] += d_total_c_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_output_kernel[IDX2C(j,h_index,Hidden_size)];
            //}

            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float ave_k = buffer[0] / Hidden_size;
			//minus ave	
			//if (tid == 0) {
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] = d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] - ave_k;
			}
            __syncthreads();                 
			
			
			//pow and sum
            buffer[tid] = 0;
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] = pow(d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)],2);
				buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
			}

			__syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float std_k = sqrt(buffer[0] / Hidden_size);
			// (x-ave)/std
			//if (tid == 0) {
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
            	d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] = d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] / std_k;
			}			
            __syncthreads();                 
			
			// x*d_layer_norm_scale + d_layer_norm_bias
			//if (tid==0) {
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] = d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_layer_norm_scale[j] + d_layer_norm_bias[j];
			}
            __syncthreads(); //!          
			
        } 
    //}
}

__global__ void residual_connection_and_norm_dec_kernel(float **d_total_h1_t, float **d_total_h_t, float **d_total_x_t, float *d_layer_norm_scale, float *d_layer_norm_bias,int Hidden_size, int s_index, int minibatch_size){
    __shared__ float buffer[256];
    //__shared__ float buffer1[256];
    const int tid = threadIdx.x;

    //for(int h_index=blockIdx.x; h_index<Hidden_size;h_index+=gridDim.x){
        for(int i=blockIdx.x;i<minibatch_size;i+=gridDim.x){
            //int s_index=i/minibatch_size;
            //int m_index=i%minibatch_size;
            int m_index = i;
			buffer[tid] = 0;
			
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] +=  d_total_x_t[s_index][IDX2C(j,m_index,Hidden_size)];  //!!
				buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
			}

            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float ave_k = buffer[0] / Hidden_size;
			//minus ave	
			//if (tid == 0) {
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] = d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] - ave_k;
			}
            __syncthreads();                 
			
			
			//pow and sum
            buffer[tid] = 0;
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)] = pow(d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)],2);
				buffer[tid] += d_total_h_t[s_index][IDX2C(j,m_index,Hidden_size)];
			}

			__syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float std_k = sqrt(buffer[0] / Hidden_size);
			// (x-ave)/std
			//if (tid == 0) {
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
            	d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] = d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] / std_k;
			}			
            __syncthreads();                 
			
			// x*d_layer_norm_scale + d_layer_norm_bias
			//if (tid==0) {
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] = d_total_h1_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_layer_norm_scale[j] + d_layer_norm_bias[j];
			}
            __syncthreads(); //!          
			
        } 
    //}
}

__global__ void residual_connection_and_norm_dec_error_kernel(float **d_total_h1_t, float **d_total_h_t, float **d_total_x_t, float *d_layer_norm_scale, float *d_layer_norm_bias,int Hidden_size, int index, int minibatch_size){
    __shared__ float buffer[256];
    const int tid = threadIdx.x;

    for(int h_index=blockIdx.x; h_index<Hidden_size;h_index+=gridDim.x){
        //for(int i=blockIdx.y;i<Sentence_len*minibatch_size;i+=gridDim.y){
        //    int s_index=i/minibatch_size;
        //    int m_index=i%minibatch_size;
			int m_index = blockIdx.y;
            buffer[tid] = 0;
			
			//if (tid==0)	{
			//	d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h_t[s_index][IDX2C(h_index,m_index,Hidden_size)] + d_total_x_t[s_index][IDX2C(h_index,m_index,Hidden_size)];
			//}
            
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				buffer[tid] += d_total_h_t[index][IDX2C(j,m_index,Hidden_size)] + d_total_x_t[index][IDX2C(j,m_index,Hidden_size)];
			}

            //for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
            //    buffer[tid] += d_total_c_t[s_index][IDX2C(j,m_index,Hidden_size)] * d_output_kernel[IDX2C(j,h_index,Hidden_size)];
            //}

            __syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float ave_k = buffer[0] / Hidden_size;
			//minus ave	
			if (tid == 0) {
				d_total_h_t[index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h_t[index][IDX2C(h_index,m_index,Hidden_size)] - ave_k;
			}
            __syncthreads();                 

			//pow and sum
            buffer[tid] = 0;
            for (int j=threadIdx.x;j<Hidden_size;j+=blockDim.x){
				buffer[tid] += pow(d_total_h_t[index][IDX2C(j,m_index,Hidden_size)],2);
			}
            
			__syncthreads();
            for(int stride = 256/2;stride > 0;stride>>=1){
                if(tid < stride){
                    buffer[tid] += buffer[stride + tid];
                }    
                __syncthreads();
            }    
            __syncthreads();
                                                                 
            float std_k = sqrt(buffer[0] / Hidden_size);
			// (x-ave)/std
			if (tid == 0) {
            	d_total_h1_t[index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h_t[index][IDX2C(h_index,m_index,Hidden_size)] / std_k;
			}			
            __syncthreads();                 
			
			// x*d_layer_norm_scale + d_layer_norm_bias
			d_total_h1_t[index][IDX2C(h_index,m_index,Hidden_size)] = d_total_h1_t[index][IDX2C(h_index,m_index,Hidden_size)] * d_layer_norm_scale[h_index] + d_layer_norm_bias[h_index];

        //} 
    }
}

/*
__global__ void lookup_pa_kernel(float **d_lookup_total, float *d_W, int *d_wids, int Hidden_size, int Sentence_len)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x; //minibatch index
	//int k = threadIdx.y; //word index
	
	if(i < Hidden_size) {// && k < Sentence_len) {	 //k: word index;
		
		d_lookup_total[j][IDX2C(i,0,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)];
	}

}
*/

__global__ void attention_weight_normalization_kernel(float **d_normal_alignments,float **d_alignments,int Hidden_size, int Sentence_len, int minibatch_size, int num_heads)
{
	int tid = threadIdx.x;
	//int tidy = threadIdx.y; //word index in a sentence
	float max_val = 0;
	float sum = 0;

	int node_index = blockIdx.x; // nodes index
	
	for(int i=0; i<Sentence_len; i++) {
		if(d_alignments[blockIdx.x][IDX2C(tid,i,minibatch_size*num_heads)] > max_val) {
			max_val = d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)];
		}	
	}
	
	for(int i=0; i<Sentence_len; i++) {
		d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] = exp(d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] - max_val);
		sum += d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)];
	}

	if(sum != 0){
		for(int i=0; i<Sentence_len; i++) {
			d_normal_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] = d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] / sum;
		}
	}
}

/*
__global__ void attention_weight_normalization_kernel_2(float **d_normal_alignments,float **d_alignments,int Hidden_size, int Sentence_len, int minibatch_size, int num_heads)
{
	int tid = threadIdx.x;
	//int tidy = threadIdx.y; //word index in a sentence
	float max_val = 0;
	float sum = 0;

	int node_index = blockIdx.x; // nodes index
	
	for(int i=b; i<Sentence_len; i++) {
		if(d_alignments[blockIdx.x][IDX2C(tid,i,minibatch_size*num_heads)] > max_val) {
			max_val = d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)];
		}	
	}
	
	for(int i=0; i<Sentence_len; i++) {
		d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] = exp(d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] - max_val);
		sum += d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)];
	}

	if(sum != 0){
		for(int i=0; i<Sentence_len; i++) {
			d_normal_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] = d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] / sum;
		}
	}
}


__global__ void attention_weight_normalization_kernel_3(float **d_normal_alignments,float **d_alignments,int Hidden_size, int Sentence_len, int minibatch_size, int num_heads)
{
	const int tid = threadIdx.x;
	int batch_heads_index = blockIdx.y; 
	float max_val = 0;
	float sum = 0;

	int node_index = blockIdx.x; // nodes index
	
	buffer[tid] = 0;
	//for(int i=blockIdx.y; i<minibatch_size*num_heads; i+=gridDim)
	for(int i=threadIdx.x; i<Sentence_len; i+=blockIdx.x){
		buffer[tid] += exp(d_alignments[node_index][IDX2C(batch_heads_index,i,minibatch_size*num_heads)]);
	}
	
	__syncthreads();
	for(int stride = 64/2;stride > 0;stride>>=1){
		if(tid < stride){
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}
	__syncthreads();
	
	float sum_k = buffer[0];   
	
	if(tid == 0) {
		d_normal_alignments[node_index][IDX2C(batch_heads_index,w_?,minibatch_size*num_heads)] = buffer[num_heads_index][0];
	}
	__syncthreads();
	

		
		d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] = exp(d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] - max_val);
		sum += d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)];
	}

	if(sum != 0){
		for(int i=0; i<Sentence_len; i++) {
			d_normal_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] = d_alignments[node_index][IDX2C(tid,i,minibatch_size*num_heads)] / sum;
		}
	}

}

*/


__global__ void lookup_kernel(float *d_lookup, float *d_W, int *d_wids, int LSTM_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x;
    if(i < LSTM_size) {
        d_lookup[IDX2C(i,j,LSTM_size)] = d_W[IDX2C(i,d_wids[j],LSTM_size)];
	}
}

__global__ void lookup_and_pos_encoding_kernel(float *d_lookup, float *d_W, int *d_wids, int Hidden_size, int w_index)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x;
	
	int half_hidden_size = Hidden_size/2;
	int num_scale = Hidden_size/2 -1;
	if (w_index != 0) {
		if(i < half_hidden_size) {
			d_lookup[IDX2C(i,j,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)] * sqrt(float(Hidden_size)) + sin(w_index/(pow(10000,float(i)/float(num_scale)))); //!! zero  
			//d_lookup[IDX2C(i,j,Hidden_size)] = d_lookup[IDX2C(i,j,Hidden_size)] + sin(w_index/(pow(10000,float(i)/float(num_scale)))); //!! zero  
		}
		else if (i<Hidden_size) {
			d_lookup[IDX2C(i,j,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)] * sqrt(float(Hidden_size)) + cos(w_index/(pow(10000,float(i-half_hidden_size)/float(num_scale))));	
			//d_lookup[IDX2C(i,j,Hidden_size)] = d_lookup[IDX2C(i,j,Hidden_size)] + cos(w_index/(pow(10000,float(i-half_hidden_size)/float(num_scale))));	
		}
	}
	else {
		if(i < half_hidden_size) {
			d_lookup[IDX2C(i,j,Hidden_size)] = sin(w_index/(pow(10000,float(i)/float(num_scale)))); //!! zero  
		}
		else if (i<Hidden_size) {
			d_lookup[IDX2C(i,j,Hidden_size)] = cos(w_index/(pow(10000,float(i-half_hidden_size)/float(num_scale))));	
		}
	
	}
}

__global__ void lookup_and_pos_encoding_2_kernel(float *d_lookup, float *d_W, int *d_wids, int Hidden_size, int w_index)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x;
	
	int half_hidden_size = Hidden_size/2;
	int num_scale = Hidden_size/2 -1;

    if(i < half_hidden_size) {
        d_lookup[IDX2C(i,j,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)] + sin(w_index/(pow(10000,float(i)/float(num_scale)))); 
		d_lookup[IDX2C(i,j,Hidden_size)] = d_lookup[IDX2C(i,j,Hidden_size)] * sqrt(float(Hidden_size));
	}
	else if (i<Hidden_size) {
        d_lookup[IDX2C(i,j,Hidden_size)] = d_W[IDX2C(i,d_wids[j],Hidden_size)] + cos(w_index/(pow(10000,float(i-half_hidden_size)/float(num_scale))));	
		d_lookup[IDX2C(i,j,Hidden_size)] = d_lookup[IDX2C(i,j,Hidden_size)] * sqrt(float(Hidden_size));
	}
}

__global__ void matrix_add_vector_kernel(float *d_h_t, float *d_bias, int Hidden_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x;
    if(i < Hidden_size) {
		d_h_t[IDX2C(i,j,Hidden_size)] = d_h_t[IDX2C(i,j,Hidden_size)] + d_bias[i];
	}
}

__global__ void matrix_relu_kernel(float *d_h_t, int Hidden_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x;
    if(i < Hidden_size) {
		if (d_h_t[IDX2C(i,j,Hidden_size)]<0) {
			d_h_t[IDX2C(i,j,Hidden_size)] = 0;
		}
	}
}

__forceinline__ __device__ float sigmoidf(float in)
{
	return 1.f / (1.f + expf(-in)); 
}

__global__ void forward_sigmoid_kernel(float *d_final,float *temp1, float *temp2, float *bias, int LSTM_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx< LSTM_size) {
	   
	   d_final[IDX2C(idx,j,LSTM_size)] = sigmoidf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + bias[idx]);
	   //d_final[idx] = sigmoidf(temp1[idx] + temp2[idx] + bias[idx]);
   }
}

__global__ void forward_sigmoid_kernel_feed(float *d_final,float *temp1, float *temp2, float *temp3, float *bias, int LSTM_size)
{	
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx< LSTM_size) {
	   d_final[IDX2C(idx,j,LSTM_size)] = sigmoidf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + temp3[IDX2C(idx,j,LSTM_size)] + bias[idx]);
   }
}

__global__ void forward_tanh_kernel(float *d_final,float *temp1, float *temp2, float *bias, int LSTM_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx < LSTM_size) {
   
	   d_final[IDX2C(idx,j,LSTM_size)] = tanhf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + bias[idx]);
   }
}

__global__ void forward_tanh_kernel_feed(float *d_final,float *temp1, float *temp2, float * temp3, float *bias, int LSTM_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx < LSTM_size) {
   
	   d_final[IDX2C(idx,j,LSTM_size)] = tanhf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + temp3[IDX2C(idx,j,LSTM_size)] + bias[idx]);
   }
}

__global__ void forward_c_t_kernel(float *d_c_t, float *d_f_t, float *d_c_t_prev, float *d_i_t, float *d_c_prime_t_tanh, int LSTM_size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(idx < LSTM_size) {
		
		d_c_t[IDX2C(idx,j,LSTM_size)] = d_f_t[IDX2C(idx,j,LSTM_size)] * d_c_t_prev[IDX2C(idx,j,LSTM_size)] + d_i_t[IDX2C(idx,j,LSTM_size)] * d_c_prime_t_tanh[IDX2C(idx,j,LSTM_size)];
		
		//d_c_t[idx] = d_f_t[idx] * d_c_t_prev[idx] + d_i_t[idx] * d_c_prime_t_tanh[idx];
		//d_c_t_store[idx] = d_c_t[idx];
	}
}

__global__ void forward_h_t_kernel(float *d_h_t, float *d_o_t, float *d_c_t, int LSTM_size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(idx < LSTM_size) {
		
		d_h_t[IDX2C(idx,j,LSTM_size)] = d_o_t[IDX2C(idx,j,LSTM_size)] * tanhf(d_c_t[IDX2C(idx,j,LSTM_size)]); //tanh
		
		//d_h_t[idx] = d_o_t[idx] * tanhf(d_c_t[idx]);
		//d_h_t_store[idx] = d_h_t[idx];
	}
}

__global__ void normalization_alignment_kernel(float *d_normal_alignment, float *d_alignment, int minibatch_size, int T) {

	int tid = threadIdx.x;
	if (tid<minibatch_size) {
		float max_val = 0;
		float sum = 0;

		for (int i=0; i<T; i++) {
			if(d_alignment[IDX2C(tid,i,minibatch_size)] > max_val) {
				max_val = d_alignment[IDX2C(tid,i,minibatch_size)];
			}
		}

		for (int i=0; i<T; i++) {
			d_alignment[IDX2C(tid,i,minibatch_size)] = exp(d_alignment[IDX2C(tid,i,minibatch_size)] - max_val); // exp --> double
			sum += d_alignment[IDX2C(tid,i,minibatch_size)];
		}
		
		if (sum != 0) {
			for (int i=0; i<T; i++) {
				d_normal_alignment[IDX2C(tid,i,minibatch_size)] = d_alignment[IDX2C(tid,i,minibatch_size)]/sum;
			}
		}
	}
}

__global__ void tanh_att_forward_kernel(float *d_output, float *d_in1, float *d_in2, float *d_bias, int LSTM_size, int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_output [i] = tanhf(d_in1[i] + d_in2[i] + d_bias[i%LSTM_size]);
	}
}

__global__ void matrix_bias_kernel(float *d_mat, float *d_vec, float *d_mat_final, int vocab_size) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(idx < vocab_size) {
		d_mat_final[IDX2C(idx,j,vocab_size)] = d_mat[IDX2C(idx,j,vocab_size)] + d_vec[idx];
	}
}

__global__ void exp_overflow_prevention(float *m, int rows){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(i<rows){
		//m[IDX2C(i,j,rows)] = expf(m[IDX2C(i,j,rows)]-15); // for prevention overflow
		m[IDX2C(i,j,rows)] = expf(m[IDX2C(i,j,rows)]-10); // for prevention overflow
		//m[IDX2C(i,j,rows)] = expf(m[IDX2C(i,j,rows)]);
	}
}

__global__ void divide(float *v1, float *v2, float *v3, int rows){
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(i<rows){
		v1[IDX2C(i,j,rows)] = v2[IDX2C(i,j,rows)]/v3[j];
	}	
}

__global__ void outputdist_overflow_prevention_kernel(float *output, float *input, int dim) {
	
	__shared__ float buffer[256]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	float *input_k = input + k*dim; //all threads in block start from same index
	float *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	buffer[threadIdx.x] = -FLT_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		float z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=256/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	float max_k = buffer[0];
	__syncthreads();
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		float z = expf(input_k[i]-max_k);
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=256/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}


  	__syncthreads();

  	// normalize the softmax
	float sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = output[i] / sum_k;
	}
}

__global__ void check_d_total_hs_mat(float **d_total_hs_mat, float *d_check, int LSTM_size,int minibatch_size,int longest_size)
{
   for(int i=blockIdx.x; i <longest_size*minibatch_size; i+=gridDim.x)
   {
       int  minibatch_index=i%minibatch_size;
       int  s_index=i/minibatch_size;
       for(int j=threadIdx.x; j < LSTM_size ;j+=blockDim.x)
       {    
           d_check[IDX2C(j,i,LSTM_size)]=d_total_hs_mat[s_index][IDX2C(j,minibatch_index,LSTM_size)];
       }    
   }
}
