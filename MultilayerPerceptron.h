/*
C++ implementation of a multilayer perceptron


Copyright (c) 2013, Ulf Biallas
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H


#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <algorithm>


class MultilayerPerceptron {
      
	struct WeightMatrix {
		int inputDim;
		int outputDim;		
		std::vector<float> w;

		WeightMatrix(int inputDim_, int outputDim_, float initialWeightScale_) {
			w.clear();
			inputDim = inputDim_;
			outputDim = outputDim_;
			for (int k=0; k<inputDim*outputDim; ++k) {
				w.push_back( 2*initialWeightScale_* (rand() / double(RAND_MAX)) - initialWeightScale_ );
			}
		}
	};


	struct Layer {
		int dim;
		std::vector<float> in;
		std::vector<float> out;
		std::vector<float> err;
		
		Layer(int dim_) {
			dim = dim_;
			for (int k=0; k<dim; ++k) {
				in.push_back(0);
				out.push_back(0);
				err.push_back(0);
			}
		}
	};
    



    public:

		struct TrainingElement {
			std::vector<float> in;
			std::vector<float> out;		
			
			TrainingElement(std::vector<float> in_, std::vector<float> out_) {
				in = in_;
				out = out_;
			}	
		};

		
		MultilayerPerceptron(int inputDimension_, int outputDimension_);
		~MultilayerPerceptron();
		
		void addHiddenLayer(int dimension_);
		void init();
		void resetWeights();
		std::vector<float> classify(std::vector<float> x_);
		void setTrainingSet(std::vector<TrainingElement> trainingSet_);
		float train(float eta_);
		
	private:
		
		float psi(float x_);
		float dpsidx(float x_);
		void calcLayerInput(int h_);
		void calcLayerOutput(int h_);
		void calcLayerError(int h_);
		void updateWeights(int h_, float eta_);
	
		int H;
		int inputDimension;
		int outputDimension;

		std::vector<WeightMatrix> weights;
		std::vector<Layer> layers;
		std::vector<TrainingElement> trainingSet;

};


#endif

