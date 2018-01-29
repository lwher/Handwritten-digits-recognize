#ifndef SJTU_LR_HPP
#define SJTU_LR_HPP

#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

class LogisticRegression {
private:
	double *W, B; // y = sigmoid(Wx + B);
	double learningRate, lossBias;
	// lossBias: consider the unbalance of training data (default: 1)
	// loss of positive sample will multiply lossBias
	// total loss = lossBias * loss_of_possitive + loss_of_negative
	int inputSize;

	double Sigmoid(double x) {
		return 1.0 / (1 + exp(-x));
	}

public:
	LogisticRegression(int _inputSize = 100, double _learningRate = 0.1, double _lossBias = 1) {
		inputSize = _inputSize;
		learningRate = _learningRate;
		lossBias = _lossBias;
		W = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			W[i] = 0;
		}
	}

	void ParameterInitialize() {
		srand(time(NULL));
		for (int i = 0; i < inputSize; ++i) {
			W[i] = 1.0 * (rand() % 201) / 100 - 1.0;
		}
		B = 1.0 * (rand() % 201) / 100 - 1.0;
	}

	void SetLearningRate(double _learningRate) {
		learningRate = _learningRate;
	}

	double TrainWithBatch(const std::vector<std::vector<double>> &data, const std::vector<int> &label) { // label: 0 or 1, return loss
		double loss = 0.0;
		double *gradsW, gradsB = 0;
		gradsW = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			gradsW[i] = 0.0;
		}
		for (int i = 0; i < data.size(); ++i) {
			double y = 0.0;
			for (int j = 0; j < inputSize; ++j) {
				y += W[j] * data[i][j];
			}
			y += B;
			y = Sigmoid(y);
			if (label[i] == 1) {
				loss += -log(y) * lossBias;
			}
			else {
				loss += -log(1 - y);
			}
			double grads = label[i] == 1 ? (1 - y) * lossBias : -y;
			for (int j = 0; j < inputSize; ++j) {
				gradsW[j] += grads * data[i][j];
			}
			gradsB += grads;
		}
		for (int j = 0; j < inputSize; ++j) {
			W[j] += learningRate * gradsW[j] / data.size();
		}
		B += learningRate * gradsB / data.size();
		delete[] gradsW;
		return loss / data.size();
	}

	double Inference(const std::vector<double> &data) {
		if (data.size() != inputSize) {
			printf("Data shape does not match while infering!\n");
			return -1;
		}
		double y = 0;
		for (int j = 0; j < inputSize; ++j) {
			y += W[j] * data[j];
		}
		y += B;
		return Sigmoid(y);
	}

	void SaveModel(const char* fileName) {
		std::ofstream fout(fileName);
		for (int i = 0; i < inputSize; ++i) {
			fout << W[i] << " ";
		}
		fout << B << std::endl;
	}

	void LoadModel(const char* fileName) {
		std::ifstream fin(fileName);
		for (int i = 0; i < inputSize; ++i) {
			fin >> W[i];
		}
		fin >> B;
	}

	~LogisticRegression() {
		delete[] W;
	}
};

#endif