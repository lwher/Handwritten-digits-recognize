#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <ctime>
#include <queue>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "BP_Neural_Networks.hpp"

const int train_data_num = 60000;
const int test_data_num = 10000;
const int hide_node = 15;
const int output_node = 10;
const int V_D = 784;
const double rate = 0.1;

using namespace std;

char read_num[5], num;

double train_data[train_data_num][V_D];
int train_label[train_data_num];

double test_data[test_data_num][V_D];
int test_label[test_data_num];

void train_data_init(){
	ifstream fin_train("train-images.idx3-ubyte", ios :: binary);
	ifstream fin_label("train-labels.idx1-ubyte", ios :: binary);
	fin_train.read(read_num, sizeof(char) * 4);
	fin_train.read(read_num, sizeof(char) * 4);
	fin_train.read(read_num, sizeof(char) * 4);
	fin_train.read(read_num, sizeof(char) * 4);
	fin_label.read(read_num, sizeof(char) * 4);
	fin_label.read(read_num, sizeof(char) * 4);
	char pix;
	for(int k = 0; k < train_data_num; ++k){
		int tot = -1;
		for(int i = 0; i < 28; ++i){
			for(int j = 0; j < 28; ++j){
				fin_train.read(&pix, sizeof(char));
				train_data[k][++tot] = (pix == 0) ? 0.0 : 1.0;
			}
		}
		fin_label.read(&num, sizeof(char));
		train_label[k] = int(num);
	}
	fin_train.close();
	fin_label.close();
}

void test_data_init(){
	ifstream fin_test("t10k-images.idx3-ubyte", ios :: binary);
	ifstream fin_label("t10k-labels.idx1-ubyte", ios :: binary);
	fin_test.read(read_num, sizeof(char) * 4);
	fin_test.read(read_num, sizeof(char) * 4);
	fin_test.read(read_num, sizeof(char) * 4);
	fin_test.read(read_num, sizeof(char) * 4);
	fin_label.read(read_num, sizeof(char) * 4);
	fin_label.read(read_num, sizeof(char) * 4);
	char pix;
	for(int k = 0; k < test_data_num; ++k){
		int tot = -1;
		for(int i = 0; i < 28; ++i){
			for(int j = 0; j < 28; ++j){
				fin_test.read(&pix, sizeof(char));
				test_data[k][++tot] =  (pix == 0) ? 0.0 : 1.0;
			}
		}
		fin_label.read(&num, sizeof(char));
		test_label[k] = int(num);
	}
	fin_test.close();
	fin_label.close();
}

void Test(BP_NNs &Net){
	int accept = 0;
	for(int i = 0; i < test_data_num; ++i){
		if(test_label[i] == Net.recognize(test_data[i]))
			accept++;
	}
	printf("total: %d.\n", test_data_num);
	printf("accept: %d.\n", accept);
	printf("rate of correct: %.3lf.\n", double(accept) / test_data_num);	
}

int main(){
	train_data_init();
	test_data_init();
	BP_NNs Net(hide_node, output_node, V_D, rate);
	//Net.read_weight("BP_NNs_weight.out");
	Net.prepare();
	Net.set_error(1e9);
	int T = 0;
	while(Net.get_error() > 1e-3){
		if(T % 5 == 0){
			Net.write_weight("BP_NNs_weight.out");
			Test(Net);
		}
		Net.set_error(0.0);
		for(int i = 0; i < train_data_num; ++i)
			Net.single_train(train_data[i], train_label[i]);
		Net.div_error(train_data_num);
		cout << ++T << " : ";
		printf("%.6lf\n", Net.get_error());
	}
	Net.write_weight("BP_NNs_weight.out");
}
