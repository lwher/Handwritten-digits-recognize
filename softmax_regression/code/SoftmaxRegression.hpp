#ifndef SJTU_SR_HPP
#define SJTU_SR_HPP

#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <fstream>

class SoftmaxRegression {
private:	
	double **W, *B; 
	double learningRate;
	int inputSize, outputSize;

public:
	SoftmaxRegression(int _inputSize = 100, int _outputSize = 10, double _learningRate = 0.1) {
		inputSize = _inputSize;
		outputSize = _outputSize;
		learningRate = _learningRate;
		W = new double*[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			W[i] = new double[outputSize];
		}
		B = new double[outputSize];
	}

	void ParameterInitialize() {
		srand(time(NULL));
		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < outputSize; ++j) {
				W[i][j] = 1.0 * (rand() % 201) / 100 - 1.0;
			}
		}
		for (int i = 0; i < outputSize; ++i) {
			B[i] = 1.0 * (rand() % 201) / 100 - 1.0;
		}
	}

	void SetLearningRate(double _learningRate) {
		learningRate = _learningRate;
	}

	double TrainWithBatch(const std::vector<std::vector<double>> &data, const std::vector<int> &label) { // label: 0 or 1, return loss
		double loss = 0.0;
		double **gradsW, *gradsB;
		// initialize
		gradsB = new double[outputSize];
		gradsW = new double*[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			gradsW[i] = new double[outputSize];
			for (int j = 0; j < outputSize; ++j) {
				gradsW[i][j] = 0;
			}
		}
		for (int i = 0; i < outputSize; ++i) {
			gradsB[i] = 0;
		}
		// forward && backforward
		double *Y = new double[outputSize];
		for (int k = 0; k < data.size(); ++k) {
			double total = 0;
			// forwarding
			for (int j = 0; j < outputSize; ++j) {
				Y[j] = 0;
				for (int i = 0; i < inputSize; ++i) {
					Y[j] += W[i][j] * data[k][i];
				}
				Y[j] += B[j];
				total += exp(Y[j]);
			}
			for (int j = 0; j < outputSize; ++j) {
				Y[j] = exp(Y[j]) / total;
				if (j == label[k]) {
					loss += -log(Y[j]);
					Y[j] -= 1;
				}
			}
			// backforwarding
			for (int j = 0; j < outputSize; ++j) {
				gradsB[j] += Y[j];
				for (int i = 0; i < inputSize; ++i) {
					gradsW[i][j] += Y[j] * data[k][i];
				}
			}
		}
		// update the gradience
		for (int j = 0; j < outputSize; ++j) {
			for (int i = 0; i < inputSize; ++i) {
				W[i][j] -= learningRate * gradsW[i][j] / data.size();
			}
			B[j] -= learningRate * gradsB[j] / data.size();
		}
		// clear the memory
		delete[] Y;
		delete[] gradsB;
		for (int i = 0; i < inputSize; ++i) {
			delete[] gradsW[i];
		}
		delete[] gradsW;
		// return the loss
		return loss / data.size();
	}

	std::vector<double> Inference(const std::vector<double> &data) {
		if (data.size() != inputSize) {
			printf("Data shape does not match while infering!\n");
			std::vector<double> emptyResult;
			return emptyResult;
		}
		std::vector<double> result(outputSize);
		double total = 0;
		for (int j = 0; j < outputSize; ++j) {
			result[j] = 0;
			for (int i = 0; i < inputSize; ++i) {
				result[j] += W[i][j] * data[i];
			}
			result[j] += B[j];
			total += exp(result[j]);
		}
		for (int j = 0; j < outputSize; ++j) {
			result[j] = exp(result[j]) / total;
		}
		return result;
	}

	void SaveModel(const char* fileName) {
		std::ofstream fout(fileName);
		for (int j = 0; j < outputSize; ++j) {
			fout << B[j] << " ";
			for (int i = 0; i < inputSize; ++i) {
				fout << W[i][j] << " ";
			}
		}
		fout << std::endl;
	}

	void LoadModel(const char* fileName) {
		std::ifstream fin(fileName);
		for (int j = 0; j < outputSize; ++j) {
			fin >> B[j];
			for (int i = 0; i < inputSize; ++i) {
				fin >> W[i][j];
			}
		}
	}

	~SoftmaxRegression() {
		delete[] B;
		for (int i = 0; i < inputSize; ++i) {
			delete[] W[i];
		}
		delete[] W;
	}
};

#endif