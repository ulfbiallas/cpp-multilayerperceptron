/*
C++ implementation of a multilayer perceptron


Copyright (c) 2013, Ulf Biallas
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#include "MultilayerPerceptron.h"

using namespace std;

MultilayerPerceptron::MultilayerPerceptron(int inputDimension_, int outputDimension_) {
	inputDimension = inputDimension_;
	outputDimension = outputDimension_;
	
	layers.push_back(Layer(inputDimension));
	H = 1;
}

MultilayerPerceptron::~MultilayerPerceptron() {

}

void MultilayerPerceptron::addHiddenLayer(int dimension_) {
	layers.push_back(Layer(dimension_));
	H++;
}

void MultilayerPerceptron::init() {
	layers.push_back(Layer(outputDimension));
	H++;
	
	resetWeights();
		
	WeightMatrix *weightMatrix;
	for(int h=0; h<H-1; ++h) {
		weightMatrix = &(weights[h]);
	}
		
}


void MultilayerPerceptron::resetWeights() {
	weights.clear();
	int h;
	int dim0, dim1;
	for(h=0; h<H-1; ++h) {
		dim0 = layers[h].dim;
		dim1 = layers[h+1].dim;
		weights.push_back(WeightMatrix(dim0, dim1, 1.0f));
	}	
}


void MultilayerPerceptron::calcLayerInput(int h_) {
	if(h_>0 && h_<H) {
		WeightMatrix *w = &(weights[h_-1]);
		int i,j;
		for(i=0; i<layers[h_].dim; ++i) {
			layers[h_].in[i] = 0;
			for(j=0; j<layers[h_-1].dim; ++j) {
				layers[h_].in[i] += layers[h_-1].out[j] * w->w[i*w->inputDim+j];
			}
		}
	}
}


void MultilayerPerceptron::calcLayerOutput(int h_) {
	for(int i=0; i<layers[h_].dim; ++i) {
		layers[h_].out[i] = psi(layers[h_].in[i]);
	}
}


vector<float> MultilayerPerceptron::classify(vector<float> x_) {
	int h;
	int i;
	if(x_.size() == inputDimension) {
		for(i=0; i<inputDimension; ++i) {
			layers[0].out[i] = x_[i];
		}
		for(h=1; h<H; ++h) {
			calcLayerInput(h);
			calcLayerOutput(h);
		}
		return layers[H-1].out;
	}
	return x_;
}


void MultilayerPerceptron::calcLayerError(int h_) {
	int i,j;
	WeightMatrix *w = &(weights[h_]);
	for(i=0; i<layers[h_].dim; ++i) {
		float sum = 0;
		for(j=0; j<layers[h_+1].dim; ++j) {
			sum += w->w[j * w->inputDim + i] * layers[h_+1].err[j];
		}
		layers[h_].err[i] = dpsidx(layers[h_].in[i]) * sum;
	}
}


void MultilayerPerceptron::updateWeights(int h_, float eta_) {
	WeightMatrix *w = &(weights[h_-1]);
	int i,j;
	float dw;
	for(i=0; i<w->outputDim; ++i) {
		for(j=0; j<w->inputDim; ++j) {
			dw = eta_ * ( layers[h_].err[i] * layers[h_-1].out[j]);
			w->w[i * w->inputDim + j] += dw;
		}
	}
}

		
// activation function 
float MultilayerPerceptron::psi(float x_) {
	float a = 0.5f;
	return 1.0f / (1+exp(-a*x_));
}


// derivative of the activation function
float MultilayerPerceptron::dpsidx(float x_) {
	return psi(x_) * (1-psi(x_));	
}


void MultilayerPerceptron::setTrainingSet(vector<TrainingElement> trainingSet_) {
	trainingSet = trainingSet_;
}


// training with the backpropagation algorithm
float MultilayerPerceptron::train(float eta_) {
	
	float trainingSetError = 0;
	int t, i, h;
	TrainingElement *te;
	for(t=0; t<trainingSet.size(); ++t) {
		te = &(trainingSet[t]);
		vector<float> x = te->in;
		vector<float> y_desired = te->out;
		vector<float> y_actual = classify(x);
		
		// calculate global error
		float err = 0;
		for(i=0; i<y_actual.size(); ++i) {
			err += pow(y_desired[i] - y_actual[i], 2);
		}
		trainingSetError += err*err;

		// calculate error in output layer (H-1)
		for(i=0; i<layers[H-1].dim; ++i) {
			layers[H-1].err[i] = y_desired[i] - y_actual[i];
		}
		
		// backpropagate the error
		for(h=H-2; h>=0; h--) {
			calcLayerError(h);
		}
		
		for(h=1; h<H; ++h) {
			updateWeights(h, eta_);
		}
	}
	
	return sqrt(trainingSetError); 
}
