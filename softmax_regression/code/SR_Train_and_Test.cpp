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

#include "SoftmaxRegression.hpp"

const int train_data_num = 60000;
const int test_data_num = 10000;
const int output_node = 10;
const int V_D = 784;
const int batch_size = 256;
const int iter = 10000;
const double learning_rate = 0.5;

using namespace std;

char read_num[5], num;

double train_data[train_data_num][V_D];
int train_label[train_data_num];

double test_data[test_data_num][V_D];
int test_label[test_data_num];

void train_data_init(){
	ifstream fin_train("train-images.idx3-ubyte", ios::binary);
	ifstream fin_label("train-labels.idx1-ubyte", ios::binary);
	fin_train.read(read_num, sizeof(char)* 4);
	fin_train.read(read_num, sizeof(char)* 4);
	fin_train.read(read_num, sizeof(char)* 4);
	fin_train.read(read_num, sizeof(char)* 4);
	fin_label.read(read_num, sizeof(char)* 4);
	fin_label.read(read_num, sizeof(char)* 4);
	char pix;
	for (int k = 0; k < train_data_num; ++k){
		int tot = -1;
		for (int i = 0; i < 28; ++i){
			for (int j = 0; j < 28; ++j){
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
	ifstream fin_test("t10k-images.idx3-ubyte", ios::binary);
	ifstream fin_label("t10k-labels.idx1-ubyte", ios::binary);
	fin_test.read(read_num, sizeof(char)* 4);
	fin_test.read(read_num, sizeof(char)* 4);
	fin_test.read(read_num, sizeof(char)* 4);
	fin_test.read(read_num, sizeof(char)* 4);
	fin_label.read(read_num, sizeof(char)* 4);
	fin_label.read(read_num, sizeof(char)* 4);
	char pix;
	for (int k = 0; k < test_data_num; ++k){
		int tot = -1;
		for (int i = 0; i < 28; ++i){
			for (int j = 0; j < 28; ++j){
				fin_test.read(&pix, sizeof(char));
				test_data[k][++tot] = (pix == 0) ? 0.0 : 1.0;
			}
		}
		fin_label.read(&num, sizeof(char));
		test_label[k] = int(num);
	}
	fin_test.close();
	fin_label.close();
}

void LoadDataForBatch(int &index, vector<vector<double>> &data, vector<int> &label) {
	data.clear();
	label.clear();
	vector<double> tmpData;
	for (int i = 0; i < batch_size; ++i) {
		tmpData.clear();
		for (int j = 0; j < V_D; ++j) {
			tmpData.push_back(train_data[index][j]);
		}
		data.push_back(tmpData);
		label.push_back(train_label[index]);
		index = (index + 1) % train_data_num;
	}
}

void Test(SoftmaxRegression &classifier){
	int accept = 0, totalNum = 0;
	vector<double> data;
	for (int i = 0; i < test_data_num; ++i){
		data.clear();
		++totalNum;
		for (int j = 0; j < V_D; ++j) {
			data.push_back(test_data[i][j]);
		}
		vector<double> result = classifier.Inference(data);
		int preb = 0;
		double score = result[0];
		for (int j = 0; j < output_node; ++j) {
			if (result[j] > score) {
				score = result[j];
				preb = j;
			}
		}
		if (preb == test_label[i]) {
			++accept;
		}
	}
	printf("total: %d.\n", totalNum);
	printf("accept: %d.\n", accept);
	printf("rate of correct: %.3lf.\n", double(accept) / totalNum);
}

void Train(SoftmaxRegression &classifier) {
	classifier.ParameterInitialize();
	vector<vector<double>> data;
	vector<int> label;
	double loss, tmpLearningRate = learning_rate;
	int index = 0;
	for (int i = 1; i <= iter; ++i) {
		if (i % 2000 == 0) {
			tmpLearningRate *= 0.7;
			classifier.SetLearningRate(tmpLearningRate);
		}
		LoadDataForBatch(index, data, label);
		loss = classifier.TrainWithBatch(data, label);
		printf("iteration: %d, loss: %.5f\n", i, loss);
	}
	classifier.SaveModel("model.srModel");
}

int main(){
	train_data_init();
	test_data_init();
	SoftmaxRegression classifier(V_D, output_node, learning_rate);
	Train(classifier);
	//classifier.LoadModel("model.srModel");
	Test(classifier);
	return 0;
}
