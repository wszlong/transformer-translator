
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <sstream>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <float.h>

#include "cuda_util.h" 
#include "util.h"
#include "gpu_info_struct.h"

//#include "ATT_IH.h"
//#include "ATT_HH.h"
#include "Input_To_Hidden_Layer.h"
#include "Hidden_To_Hidden_Layer.h"
#include "Softmax_Layer.h"
//#include "ATT_IH.h"
#include "decoder.h"

#include "decoder.hpp"
//#include "ATT_IH.hpp"
//#include "ATT_HH.hpp"
#include "Input_To_Hidden_Layer.hpp"
#include "Hidden_To_Hidden_Layer.hpp"
#include "Softmax_Layer.hpp"

//Boost
#include "boost/program_options.hpp" 

using namespace std;

void command_line_parse(global_params &params, int argc, char **argv) {

	namespace po = boost::program_options;
	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Run to get help on how to use the program.")
		("model-file,m", po::value<string> (&params.model_file), "model file for test")
		("vocab-file,v", po::value<string> (&params.vocab_file), "vocab file for test")
		("small-vocab-file,s", po::value<string> (&params.small_vocab_file), "vocab file for test")
		("beam-size,b", po::value<int> (&params.beam_size), "beamsearch size. Default: 12")
		("gpu-num,g",po::value<int> (&params.gpu_num), "gpu num for test. Default: gpu 0")
		("input-test-file,i",po::value<string> (&params.test_file), "test input file")
		("output-test-file,o", po::value<string> (&params.output_file), "test output file. Default: output.txt");
	po::variables_map vm;
	try{
		
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if(vm.count("help")) {
			cout<<"****************************************************************************"<<endl;
			cout<<"******************Cuda Based C++ Code for NMT Decode Process****************"<<endl;
			cout<<desc<<endl;
			exit(EXIT_FAILURE);
		}

		if( !(vm.count("model-file") && vm.count("vocab-file") && vm.count("input-test-file")) ) {
			cout<<"Error: you must input model file and test file, '-h' for help"<<endl;
			//exit(EXIT_FAILURE);
		}



	}
	catch(po::error& e) {
		cout<<"ERROR: "<<e.what()<<endl;
		exit(EXIT_FAILURE);
	}
}


int main(int argc, char **argv)
{
	chrono::time_point<chrono::system_clock> start_total_all, start_total_decode, end_total;
	chrono::duration<double> elapsed_seconds_all, elapsed_seconds_decode;
	
	start_total_all = chrono::system_clock::now(); 
	
	global_params params;

	command_line_parse(params, argc, argv);

    ifstream ftest(params.test_file);
		
    Decoder d(params.model_file, params.vocab_file, params.small_vocab_file, params.beam_size, params.gpu_num); //model file, beam_size, gpu_num;
	
	start_total_decode = std::chrono::system_clock::now();
	
	ofstream file_output;
	file_output.open(params.output_file.c_str());
	
	cout<<"Start Translate..."<<endl;
    string s;
	int count = 1;
    while(getline(ftest,s))
    {
		cout<<"source: "<<s<<endl;
		string final = d.translate(s);
		file_output<<final<<"\n";
		cout<<"trans: "<<final<<endl;
		if(count%10 == 0) {
			cout<<count<<endl;
		}
		count++;
		//string a_test;
		//cin>>a_test;
		
    }
	cout<<"final"<<endl;
	
	end_total = chrono::system_clock::now();
	
	elapsed_seconds_decode = end_total-start_total_decode;
	cout<<"decode time: "<<(double)elapsed_seconds_decode.count()<<" seconds"<<endl;
	
	elapsed_seconds_all = end_total-start_total_all;
	cout<<"Total Program Runtime: "<<(double)elapsed_seconds_all.count()<<" seconds"<<endl;

}
