

Decoder::Decoder(string model_file, string vocab_file, string small_vocab_file, int beam_size, int gpu_num)
{
	
	this->gpu_num = gpu_num;
    this->beam_size = beam_size; //beamsearch sizei
	this->input_weight_file = model_file;
	this->input_vocab_file = vocab_file;
	this->small_vocab_file = small_vocab_file;
	if (this->small_vocab_file != "") {
		this->is_small_vocab = true;
	}
	cout<<"is_small_vocab: "<<is_small_vocab<<endl;
    
	this->B = 1;
	//this->B = beam_size;
    Tmax = 200;
    T = 200;
	
	//model params and dictionary !!!
	cout<<"Load Model and Init..."<<endl;;
	init_and_load_model(src_w2i,tgt_i2w);

}

void Decoder::init_and_load_model(map<string, int> &src_w2i, map<int, string> &tgt_i2w)
{
	
	//load vocab table and small vocab
	load_vocab();	

	//alloc params memory
	init_model_alloc();
	
	//model structure
	creat_model_structure();
	
	//alloc params memory
	//init_model_alloc();

	//load model params
	load_model_params();

}

void Decoder::load_vocab() {

	ifstream input_vocab;
	input_vocab.open(input_vocab_file.c_str());
	
	//cout<<"input_vocab_file: "<<input_vocab_file<<endl;	
	string str;
	string word;
	vector<string> file_model_info;

	getline(input_vocab, str); //model info   // 6 512 409600 40960
	istringstream iss(str, istringstream::in);
	while(iss >> word){
		file_model_info.push_back(word);
	}
	//cout<<"load_vocab test "<<endl;	
	
	num_layers_src = stoi(file_model_info[0]); 
	num_layers_tgt = stoi(file_model_info[1]); 
	Hidden_size = stoi(file_model_info[2]); //Hidden_size
	num_heads = stoi(file_model_info[3]); //num_heads
	source_vocab_size = stoi(file_model_info[4]); // 40960
	target_vocab_size = stoi(file_model_info[5]); // 40960
	
	cout<<"Hidden_size: "<<Hidden_size<<endl;	
	cout<<"num_heads: "<<num_heads<<endl;	

	getline(input_vocab, str); // ======
	
	//source dict and target dict
	while(getline(input_vocab, str)){
		int tmp_index;
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='='){
			break;	
		} 

		istringstream iss(str, istringstream::in);
		iss >> word;
		tmp_index = stoi(word);
		iss >> word;
		src_w2i[word] = tmp_index;
	}
	cout<<"src_w2i test: "<<src_w2i["中国"]<<endl;
	while(getline(input_vocab, str)){
		int tmp_index;	
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='='){
			break;
		}

		istringstream iss(str, istringstream::in);
		iss >> word;
		tmp_index = stoi(word);
		iss >> word;
		tgt_i2w[tmp_index] = word;
	}
	
	input_vocab.close();

	//for small vocab
	if (is_small_vocab) {
		cout<<"load small_vocab_file: "<<endl;	
		ifstream fs2t;
		fs2t.open(small_vocab_file.c_str());
		string s;
		while(getline(fs2t,s)) {
			stringstream ss;
			ss << s;
			int c, e;
			vector<int> es;
			ss >> c;
			while (ss >> e) {
				es.push_back(e);
			}
			s2t[c] = es;
		}
		fs2t.close();
	}

}

void Decoder::creat_model_structure() {
	
	//void init_upper_transfer_layer(bool upper_softmax,bool copy_h_t,bool source_side,Softmax_Layer *softmax,Hidden_To_Hidden_Layer *hidden_layer) {
	input_layer_source.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]);
	
	input_layer_target.upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[0]);

	for(int i=0; i<target_hidden_layers.size(); i++) {	
		//lower transfer stuff
		if(i==0) {
	//void init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer *input_layer,Hidden_To_Hidden_Layer *hidden_layer)
			//source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source,NULL); 
			target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_target,NULL);
		}
		else {
			//source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers[i-1]);
			target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i-1]);
		}
	
		//upper transfer stuff
		if(i==target_hidden_layers.size()-1) {
			//source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,true,&softmax_layer_target,NULL);
			target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,false,&softmax_layer_target,NULL);
			
			//cout<<"test source: "<<source_hidden_layers[i].upper_layer.source_side<<endl;
			//cout<<"test taget: "<<target_hidden_layers[i].upper_layer.source_side<<endl;

			softmax_layer_target.lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i]); //softmax	
		}
		else {
			//source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[i+1]);
			target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[i+1]);
		}
	}
	//for source separate because different layers
	for(int i=0; i<source_hidden_layers.size(); i++) {	
		//lower transfer stuff
		if(i==0) {
			source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source,NULL); 
		}
		else {
			source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers[i-1]);
		}
	
		//upper transfer stuff
		if(i==source_hidden_layers.size()-1) {
			source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,true,&softmax_layer_target,NULL);
		}
		else {
			source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[i+1]);
		}
	}

}


void Decoder::init_model_alloc() {
	
	//model param init
	
	//for debug
	//num_layers = 6;
	//Hidden_size = 512;
	//target_vocab_size = 40960;
	//source_vocab_size = 40960;
	//feed_input = true;
	
	
	//model layers structure
	for(int i=0; i<num_layers_src-1; i++){
		source_hidden_layers.push_back(Hidden_To_Hidden_Layer());
	}
	for(int i=0; i<num_layers_tgt-1; i++){
		target_hidden_layers.push_back(Hidden_To_Hidden_Layer());
	}

	input_layer_source.init_Input_To_Hidden_Layer(Hidden_size, Hidden_size, 1, source_vocab_size, false, gpu_num, num_heads, false, this);
		
	for(int i=0; i<num_layers_src-1; i++){
		source_hidden_layers[i].init_Hidden_To_Hidden_Layer(Hidden_size, 1, gpu_num, num_heads, false, num_layers_tgt, this);
	}
	
	
	input_layer_target.init_Input_To_Hidden_Layer(Hidden_size, Hidden_size, beam_size, target_vocab_size, true, gpu_num, num_heads, true, this);
	
	for(int i=0; i<num_layers_tgt-1; i++){
		target_hidden_layers[i].init_Hidden_To_Hidden_Layer(Hidden_size, beam_size, gpu_num, num_heads, true, num_layers_tgt, this);
	}

	softmax_layer_target.init_Softmax_Layer(Hidden_size, beam_size, target_vocab_size, gpu_num);

}

void Decoder::load_model_params() {

	//cout <<"vocab file load end, start load weight file.."<<endl;	
	ifstream input_model;
	input_model.open(input_weight_file, ios_base::in | ios_base::binary);
	//input_model.open("../model/model.bin", ios_base::in | ios_base::binary);
	//cout<<"input_weight_file: "<<input_weight_file<<endl;

	//source params	
	input_layer_source.load_weight(input_model);
	for(int i=0; i<num_layers_src-1; i++){
		//cout<< "loda source_hidden_layers: "<< i<<endl;
		source_hidden_layers[i].load_weight(input_model);	
	}
	

	//target params
	//copy encoder's d_W to decoder's d_W
	cudaSetDevice(gpu_num);
	cudaMemcpy(input_layer_target.d_W, input_layer_source.d_W, Hidden_size*source_vocab_size*sizeof(float), cudaMemcpyDeviceToDevice);

	input_layer_target.load_weight(input_model);
	for(int i=0; i<num_layers_tgt-1; i++){
		//cout<< "loda target_hidden_layers: "<< i<<endl;
		target_hidden_layers[i].load_weight(input_model);	
	}

	// attention and softmax params
	//attention_softmax_target.load_weight(input_model);
	
	cudaSetDevice(gpu_num);
	cudaMemcpy(softmax_layer_target.d_D, input_layer_source.d_W, Hidden_size*source_vocab_size*sizeof(float), cudaMemcpyDeviceToDevice);
   	
	input_model.close();

}

vector<int> Decoder::w2id(string input_sen) {

    stringstream ss;
    ss << input_sen;
    string w;
    vector<int> wids;
	int source_len = src_w2i.size();
	
	set<int> se;
	if (is_small_vocab) {
		for (int i=0; i<3; i++) {
			se.insert(i);
		}
		//se.insert(29980);
	}	
	
	while (ss>>w)	
	{
		int cid;
		if (src_w2i.find(w) != src_w2i.end() && src_w2i[w] < source_vocab_size) { //
			wids.push_back(src_w2i[w]);			
			if (is_small_vocab) {
				cid = src_w2i[w];
				for(int i=0; i<s2t[cid].size(); i++) {
					se.insert(s2t[cid][i]);
				}
			}
			//cout<<endl;
		}
		else {
			wids.push_back(2); //<UNK>
		}
	}
	wids.push_back(1); //put <eos> in the end
	
	if(is_small_vocab) {	
		tgt_vocab.clear();
		for (set<int>::iterator it=se.begin(); it!=se.end(); it++) {
			tgt_vocab.push_back(*it);
		}

		//cout<<"tgt_vocab size: "<<tgt_vocab.size()<<endl;
		//for (int i=0; i<tgt_vocab.size(); i++) {                                                                                                           
		//	cout<<tgt_vocab[i]<<" ";
		//}
		//cout<<endl;

		//for small vocab
		input_layer_target.init_small_vocab(tgt_vocab);
		softmax_layer_target.init_small_vocab(tgt_vocab);
	
	}
	//cin>>a_test;

    return wids;
	
}


string Decoder::id2w(vector<int> output_wids)
{
    string output_sen;
	if (is_small_vocab) {
		for (int i=0;i<output_wids.size() - 1;i++) {
			//output_sen += tgt_i2w[output_wids[i]] + " ";
			output_sen += tgt_i2w[tgt_vocab[output_wids[i]]] + " ";
    	}
	}
	else {
		for (int i=0;i<output_wids.size() - 1;i++) {
			output_sen += tgt_i2w[output_wids[i]] + " ";
			//output_sen += tgt_i2w[tgt_vocab[output_wids[i]]] + " ";
		}	
	}
    return output_sen;
}

void Decoder::encode(vector<int> input_wids) {
	
	B = 1; //	
	T = input_wids.size();
	
	//look-up for source side
	input_layer_source.look_up_gpu(&input_wids[0], T);
		

	input_layer_source.multi_head_attention(T);
	input_layer_source.add_and_norm(T);
	
	input_layer_source.feed_foward(T);
	input_layer_source.add_and_norm_1(T);
	
	//cout<<"encoder show d_qkv_kernel: "<<endl;
	//show_matrix(source_hidden_layers[0].d_qkv_kernel,Hidden_size, 3*Hidden_size);
	
	//cout<<"encoder show d_qkv_kernel source_hidden_layers[1]: "<<endl;
	//show_matrix(source_hidden_layers[1].d_qkv_kernel,Hidden_size, 3*Hidden_size);

	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();

	//source_hidden_layers[0].multi_head_attention(T);
	for (int i=0; i<num_layers_src-1; i++) {
	//for (int i=0; i<1; i++) {
		
		//cout<<"hidden layers: "<<i<<endl;		
		source_hidden_layers[i].multi_head_attention(T);
		source_hidden_layers[i].add_and_norm(T);
		
		source_hidden_layers[i].feed_foward(T);
		source_hidden_layers[i].add_and_norm_1(T);
		//cin>>a_test;				
	}
	


	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();

}


	
void Decoder::get_next_prob(vector<int> tgt_wids, int index, vector<int> father_idx) {
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
	
	//cout<<"B: "<<B<<endl;
	//chrono::time_point<chrono::system_clock> time0, time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11, time_h1, time_h2, time_h3, time_h4, time_h5, time_h6, time8_;
	//chrono::duration<double> elapsed_1, elapsed_2, elapsed_3, elapsed_4, elapsed_5, elapsed_6, elapsed_7, elapsed_8, elapsed_9, elapsed_10, elapsed_11, elapsed_12, elapsed_h1, elapsed_h2, elapsed_h3, elapsed_h4, elapsed_h5, elapsed_h6;

	if(index>0) { //update d_total_wx_t
		input_layer_target.update_history_state(&father_idx[0], index, B);	
		for(int i=0; i<num_layers_tgt-1; i++) {
			target_hidden_layers[i].update_history_state(&father_idx[0], index, B);
		}
	}

	input_layer_target.look_up_gpu_decoder(&tgt_wids[0], index, B, is_small_vocab);
	
	//cudaSetDevice(gpu_num);
	//cudaDeviceSynchronize();
	//time0 = chrono::system_clock::now(); 
	input_layer_target.multi_head_att_dec(index, B);
	//cudaSetDevice(gpu_num);
	//cudaDeviceSynchronize();
	//time1 = chrono::system_clock::now(); 
	input_layer_target.add_and_norm_dec(index, B);

	input_layer_target.multi_head_att_encdec(index, B, T);
	input_layer_target.add_and_norm_1_dec(index, B);
	
	input_layer_target.feed_foward_dec(index, B);
	input_layer_target.add_and_norm_2_dec(index, B);
	
	//elapsed_1 = time1 - time0;
	//cout<<"update_history_state multi_head_att_dec i2h Runtime: "<<(double)elapsed_1.count()<<" seconds"<<endl;
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
	
	for (int i=0; i<num_layers_tgt-1; i++) {
	//for (int i=0; i<1; i++) {
		//cout<<"decoder layers: "<<i<<endl;

		//time2 = chrono::system_clock::now(); 
		target_hidden_layers[i].multi_head_att_dec(index, B);
		//time3 = chrono::system_clock::now(); 
		//elapsed_2 = time3 - time2;
		//cout<<"update_history_state ann h2h Runtime: "<<(double)elapsed_2.count()<<" seconds"<<endl;
		target_hidden_layers[i].add_and_norm_dec(index, B);

		target_hidden_layers[i].multi_head_att_encdec(index, B, T);
		target_hidden_layers[i].add_and_norm_1_dec(index, B);
		
		target_hidden_layers[i].feed_foward_dec(index, B);
		target_hidden_layers[i].add_and_norm_2_dec(index, B);
		
		//cin>>a_test;	
		cudaSetDevice(gpu_num);
		cudaDeviceSynchronize();
	}
	
	//cout<<"begin softmax: "<<endl;
	softmax_layer_target.softmax_forward_prop(index, B, is_small_vocab);
	//attention_softmax_target.attention_softmax_forward_prop(T,B); // T: source length, B: current beam size
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();

}

void Decoder::generate_new_samples(vector<vector<int>> &hyp_samples, vector<float> &hyp_scores,
		vector<vector<int>> &final_samples, vector<float> &final_scores, int &dead_k, vector<int> &tgt_wids, vector<int> &father_idx) {
	
	int live_k = beam_size - dead_k;
	
	int VV;
	if (is_small_vocab) {
		VV = tgt_vocab.size();
	}
	else {
		VV = target_vocab_size;
	}
	
	vector<float> logit_softmax;
	logit_softmax.resize(VV*beam_size);
	cudaMemcpy(&logit_softmax[0], softmax_layer_target.d_logit_softmax, VV*B*sizeof(float), cudaMemcpyDeviceToHost);
	priority_queue<pair<float,pair<int,int> >,vector<pair<float,pair<int,int> > >,greater<pair<float,pair<int,int> > > > q;

	for (int i=0;i<B;i++){
		//for (int j=0;j<target_vocab_size;j++) {
		for (int j=0;j<VV;j++) {
			float score = log(logit_softmax[IDX2C(j,i,VV)]) + hyp_scores[i]; // (target_vocab_size,B)
			if (q.size() < live_k )
				q.push(make_pair(score, make_pair(i,j)));
			else {
				if (q.top().first < score) {
					q.pop();  // discard small
					q.push(make_pair(score, make_pair(i,j)));
				}
			}

		}

	}

	vector<vector<int>> new_hyp_samples;
	vector<float> new_hyp_scores;
	father_idx.clear(); //
	tgt_wids.clear();

	for(int k=0; k<live_k; k++) {
	
		float score = q.top().first; // q small -> big

		int i = q.top().second.first;
		int j = q.top().second.second;
		vector<int> sample(hyp_samples[i]); //
		sample.push_back(j);
		if(j==1) {
			dead_k += 1;
			final_samples.push_back(sample);
			float lp = pow((5+sample.size()),0.2)/pow(6,0.2);
			float score_my = score/lp;
			final_scores.push_back(score_my);
		}
		else {
			new_hyp_samples.push_back(sample);
			new_hyp_scores.push_back(score);
			tgt_wids.push_back(j);
			father_idx.push_back(i);
		}
		q.pop();
	}
/*
	cout<<"tgt_wids: "<<endl;
	for (int i=0; i<tgt_wids.size(); i++) {
		cout<<tgt_wids[i]<<" ";
	}
	cout<<endl;
	
	cout<<"father_idx: "<<endl;
	for (int i=0; i<father_idx.size(); i++) {
		cout<<father_idx[i]<<" ";
	}
	cout<<endl;
*/
	hyp_samples.swap(new_hyp_samples);
	hyp_scores.swap(new_hyp_scores);
}


vector<int> Decoder::decode() {
	
	vector<vector<int>> final_samples;
	vector<float> final_scores;
	vector<vector<int>> hyp_samples(1,vector<int>());
	vector<float> hyp_scores(1,0.0);
	int dead_k = 0;

	vector<int> tgt_wids;
	tgt_wids.push_back(0);

	//for(int i=0; i<Tmax; i++) {
	for(int i=0; i<Tmax; i++) {
		get_next_prob(tgt_wids, i, father_idx);
		generate_new_samples(hyp_samples, hyp_scores, final_samples, final_scores, dead_k, tgt_wids, father_idx);
		//if (i==1) {
		//	cin>>a_test;
		//}

		B = beam_size-dead_k;
		if(B<=0) {
			break;
		}
	}

	if(B>0) {
		for(int k=0; k<B; k++) {
			final_samples.push_back(hyp_samples[k]);
			final_scores.push_back(hyp_scores[k]);
		}
	}

	float best_score = -9999;
	int best_k = 0;
	for(int k=0; k<final_samples.size(); k++) {
		float score = final_scores[k]/final_samples[k].size();
		if(score>best_score) {
			best_score = score;
			best_k = k;
		}
	}
	
	//vector<int> a;
	//return a;
	return final_samples[best_k];
}

string Decoder::translate(string input_sen) {

	vector<int> input_wids = w2id(input_sen);
	//for(int i=0; i<input_wids.size(); i++) {
	//	cout<<input_wids[i]<<" ";
	//}

	//cout<<endl;
	
	//cout<<"encode: "<<endl;
	encode(input_wids);

	//cout<<"decode: "<<endl;
	vector<int> output_wids = decode();

	string output_sen = id2w(output_wids);

	//string s = "abc";
	return output_sen;
}





