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

