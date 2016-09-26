#ifndef SJTU_BPNNS_HPP
#define SJTU_BPNNS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstdlib>

class BP_NNs
{
private:

	double f(double x){
		return 1.00 / (1.00 + exp(x));
	}

public:
	int hide_node, output_node, V_D;

	double rate, **train_W, **train_O, *hide_X, *output_Y, error, *D, *D_output, *D_hide, *hide_lean, *output_lean;
	/*
		train_W, train_O: Net weigth
		hide_lean, output_lean: point's lean weight
		rate: rate of learning
		error: average error
		D: expected result
		D_output, D_hide: point's error 
		hide_X, output_Y: point's result
	*/

	BP_NNs(int _hidenode = 15, int _output_node = 10, int _VD = 100, double _rate = 0.3) : 
		hide_node(_hidenode), 
		output_node(_output_node), 
		V_D(_VD),
		rate(_rate){
			train_W = new double* [hide_node];
			for(int i = 0; i < hide_node; ++i) train_W[i] = new double [V_D];
			train_O = new double* [output_node];
			for(int i = 0; i < output_node; ++i) train_O[i] = new double [hide_node];
			hide_X = new double [hide_node];
			output_Y = new double [output_node];
			D = new double [output_node];
			D_hide = new double [hide_node];
			D_output = new double [output_node];
			hide_lean = new double [hide_node];
			output_lean = new double [output_node];
		}

	double get_error(){
		return error;
	}

	void set_error(double X){
		error = X;
	}
	
	void div_error(int X){
		error /= 1.00 * X;
	}

	void prepare(){

		srand(time(NULL));

		for(int i = 0; i < hide_node; ++i){
			for(int j = 0; j < V_D; ++j){
				train_W[i][j] = double(rand() % 201) / 100.0 - 1.0;
			}
		}

		for(int i = 0; i < output_node; ++i){
			for(int j = 0; j < hide_node; ++j){
				train_O[i][j] = double(rand() % 201) / 100.0 - 1.0;
			}
		}

		for(int i = 0; i < hide_node; ++i)
			hide_lean[i] = double(rand() % 201) / 100.0 - 1.0;

		for(int j = 0; j < output_node; ++j)
			output_lean[j] = double(rand() % 201) / 100.0 - 1.0;

	}

	void single_train(double *train_data, int train_label){

		for(int i = 0; i < output_node; ++i) 
			D[i] = 0.0;
		D[train_label] = 1.0;
			
		for(int i = 0; i < hide_node; ++i){
			hide_X[i] = 0;
			for(int j = 0; j < V_D; ++j)
				hide_X[i] += train_data[j] * train_W[i][j];
			hide_X[i] = f(-hide_X[i]-hide_lean[i]);
		}
			
		for(int i = 0; i < output_node; ++i){
			output_Y[i] = 0;
			for(int j = 0; j < hide_node; ++j)
				output_Y[i] += hide_X[j] * train_O[i][j];
			output_Y[i] = f(-output_Y[i]-output_lean[i]);
		}

		for(int i = 0; i < output_node; ++i){
			D_output[i] = (D[i] - output_Y[i]) * output_Y[i] * (1.0 - output_Y[i]);
			for(int j = 0; j < hide_node; ++j)
				train_O[i][j] += rate * D_output[i] * hide_X[j]; 
		}

		for(int i = 0; i < hide_node; ++i){
			D_hide[i] = 0.0;
			for(int j = 0; j < output_node; ++j)
				D_hide[i] += D_output[j] * train_O[j][i];
			D_hide[i] = D_hide[i] * hide_X[i] * (1.0 - hide_X[i]);
			for(int j = 0; j < V_D; ++j)
				train_W[i][j] += rate * D_hide[i] * train_data[j];
		}

		for(int i = 0; i < hide_node; ++i)
			hide_lean[i] += rate * D_hide[i];

		for(int i = 0; i < output_node; ++i)
			output_lean[i] += rate * D_output[i];

		for(int i = 0; i < output_node; ++i)
			error += 0.5 * (D[i] - output_Y[i]) * (D[i] - output_Y[i]);

	}

	int recognize(double *V){

		for(int i = 0; i < hide_node; ++i){
			hide_X[i] = 0;
			for(int j = 0; j < V_D; ++j)
				hide_X[i] += V[j] * train_W[i][j];
			hide_X[i] = f(-hide_X[i]-hide_lean[i]);
		}
				
		for(int i = 0; i < output_node; ++i){
			output_Y[i] = 0;
			for(int j = 0; j < hide_node; ++j)
				output_Y[i] += hide_X[j] * train_O[i][j];
			output_Y[i] = f(-output_Y[i]-output_lean[i]);
		}

		double maxP = -1;
		int res;
		
		for(int i = 0; i < output_node; ++i){
			if(output_Y[i] > maxP){
				maxP = output_Y[i];
				res = i;
			}
		}
		return res;
	}

	void read_weight(const char* argv){

		std :: ifstream fin(argv);
	
		for(int i = 0; i < hide_node; ++i){
			for(int j = 0; j < V_D; ++j)
				fin >> train_W[i][j];
		}
		
		for(int i = 0; i < output_node; ++i){
			for(int j = 0; j < hide_node; ++j)
				fin >> train_O[i][j];
		}
		
		for(int i = 0; i < hide_node; ++i)
			fin >> hide_lean[i];

		for(int i = 0; i < output_node; ++i)
			fin >> output_lean[i];

		fin.close();
	}

	void write_weight(const char* argv){

		std :: ofstream fout(argv);

		for(int i = 0; i < hide_node; ++i){
			for(int j = 0; j < V_D; ++j)
				fout << train_W[i][j] << " ";
		}
		
		for(int i = 0; i < output_node; ++i){
			for(int j = 0; j < hide_node; ++j)
				fout << train_O[i][j] << " ";
		}
		
		for(int i = 0; i < hide_node; ++i)
			fout << hide_lean[i] << " ";

		for(int i = 0; i < output_node; ++i)
			fout << output_lean[i] << " ";
		
		fout.close();
	}

	~BP_NNs(){
		for(int i = 0; i < hide_node; ++i) delete [] train_W[i];
		for(int i = 0; i < output_node; ++i) delete [] train_O[i];
		delete [] train_W;
		delete [] train_O;
		delete [] hide_X;
		delete [] hide_lean;
		delete [] output_Y;
		delete [] output_lean;
		delete [] D;
		delete [] D_hide;
		delete [] D_output;	
	};
	
};

#endif
