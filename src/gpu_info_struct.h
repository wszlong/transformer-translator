
struct gpu_info_struct {
	int device_number;
	cublasHandle_t handle;

	cudaStream_t s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10, s1_feed,s2_feed,s3_feed,s4_feed;

	cudaEvent_t e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10, e1_feed,e2_feed,e3_feed,e4_feed;
	
	cudaEvent_t h_t_below_transfer;

	void init(int device_number) {
	
		this->device_number = device_number;
		cudaSetDevice(device_number);
		cublasCreate(&handle);

		cudaStreamCreate(&s0);
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);
		cudaStreamCreate(&s4);
		cudaStreamCreate(&s5);
		cudaStreamCreate(&s6);
		cudaStreamCreate(&s7);
		cudaStreamCreate(&s8);
		cudaStreamCreate(&s9);
		cudaStreamCreate(&s10);
		
		cudaStreamCreate(&s1_feed);
		cudaStreamCreate(&s2_feed);
		cudaStreamCreate(&s3_feed);
		cudaStreamCreate(&s4_feed);

		cudaEventCreate(&e0);
		cudaEventCreate(&e1);
		cudaEventCreate(&e2);
		cudaEventCreate(&e3);
		cudaEventCreate(&e4);
		cudaEventCreate(&e5);
		cudaEventCreate(&e6);
		cudaEventCreate(&e7);
		cudaEventCreate(&e8);
		cudaEventCreate(&e9);
		cudaEventCreate(&e10);
		
		cudaEventCreate(&e1_feed);
		cudaEventCreate(&e2_feed);
		cudaEventCreate(&e3_feed);
		cudaEventCreate(&e4_feed);
		
		cudaEventCreate(&h_t_below_transfer);
	
	}	
};
