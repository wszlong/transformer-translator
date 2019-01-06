
#include <fstream>
#include <string>
#include <sstream>
#include <fstream>
#include <cublas_v2.h>
using namespace std;

#include <vector>

struct global_params {

	int gpu_num = 0;
	int beam_size = 12;
	string model_file;
	string vocab_file;
	string small_vocab_file = "";
	string test_file;
	string output_file = "output.txt";
};

void show_matrix(float *d_m, int r, int c)
{
    vector<float> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i=0;i<r;i++)
    for (int i=0;i<min(r,10);i++)
    {
        //for (int j=0;j<c;j++)
        for (int j=0;j<min(c,12);j++)
        {
            cout<<m[IDX2C(i,j,r)]<<' ';
        }
        cout<<endl;
    }
}

void show_matrix_lz(float *d_m, int r, int c)
{
    vector<float> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i=0;i<r;i++)
    for (int i=1024;i<1034;i++)
    {
        //for (int j=0;j<c;j++)
        for (int j=0;j<min(c,12);j++)
        {
            cout<<m[IDX2C(i,j,r)]<<' ';
        }
        cout<<endl;
    }
}

void show_matrix_test(float *d_m, int r, int c)
{
    vector<float> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i=0;i<r;i++)
    //for (int i=0;i<10;i++)
    {
        for (int j=0;j<c;j++)
		//for (int j=0;j<min(c,12);j++)
        //for (int j=2560;j<2560+12;j++)
        {
            cout<<m[IDX2C(i,j,r)]<<' ';
        }
        cout<<endl;
    }
}

void show_matrix(int *d_m, int r, int c)
{
    vector<int> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i=0;i<r;i++)
    for (int i=0;i<min(10,r);i++)
    {
        //for (int j=0;j<c;j++)
        for (int j=0;j<min(c,12);j++)
        {
            cout<<m[IDX2C(i,j,r)]<<' ';
        }
        cout<<endl;
    }
}

/***
void read_matrix_GPU(float *d_mat, int rows, int cols, ifstream &input) 
{
	float *temp_mat = (float *)malloc(rows*cols*sizeof(float));
	string temp_string;
	string temp_token;

	for(int i=0; i<rows; i++){
		getline(input, temp_string);
		istringstream iss_input(temp_string, istringstream::in);

		for(int j=0; j<cols; j++){
			iss_input >> temp_token;
			temp_mat[IDX2C(i,j,rows)] = stod(temp_token);
		}
	}

	getline(input,temp_string); // blank line
	cudaMemcpy(d_mat,temp_mat,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
	free(temp_mat);
}
***/

void read_matrix_GPU(float *d_mat, int rows, int cols, ifstream &input) 
{
	vector<float> v;
	v.resize(rows*cols);
	input.read((char*)&v[0],sizeof(float)*rows*cols);
	cudaMemcpy(d_mat,&v[0],rows*cols*sizeof(float),cudaMemcpyHostToDevice);

}


std::string cublasErrorString(cublasStatus_t error) {
	
	switch (error) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

void CUBLAS_ERROR_WRAPPER(cublasStatus_t cudaStat,std::string error_message) 
{	
	if (cudaStat != CUBLAS_STATUS_SUCCESS) {
		std::string msg = cublasErrorString(cudaStat);
		std::cout << error_message << std::endl;
		cout << msg << endl;
		//exit (EXIT_FAILURE);
	}
}

void CUDA_GET_LAST_ERROR(std::string msg) {
	cudaError_t code = cudaGetLastError();
	if ( cudaSuccess != code ) {
		std::cout << "Error in kernel\n";
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
		std::cout << msg << "\n";
		//exit (EXIT_FAILURE);
	}
}
