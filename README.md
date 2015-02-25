
# multilayer perceptron

C++ implementation of a multilayer perceptron.



### API ###

	MultilayerPerceptron(int inputDimension, int outputDimension);
	
Creates a new MultilayerPerceptron with the given input and output dimension.


	void addHiddenLayer(int dimension);

Adds a further hidden layer with the given dimension.


	void init();

Initializes the MLP after adding the desired number of hidden layers.


	void resetWeights();

Resets the MLP.


	std::vector<float> classify(std::vector<float> x);

Classifies a vector of the input dimension.
The returned vector contains in each component the probability that the input vector belongs to the associated class.


	void setTrainingSet(std::vector<TrainingElement> trainingSet);

Defines a training set as a vector of TrainingElement's. Each TrainingElement maps an input vector onto an output vector.


	float train(float eta);

Performs one training step with the given learning rate using the backpropagation algorithm.


	
### The Demo ###

You can find a demo project in the repository which is using the library. To compile it, you will need OpenGL and GLUT.
In this demo, you can place points on a plane in three different colors. 
Afterwards you can train the MLP with your points as training set. 
During the training you can observe how the different points in the plane are mapped by the MLP to one of the colors
and how the mapping gets better and better due to the training.

#### Keys ####

[1] set point color to red

[2] set point color to green

[3] set point color to blue

[click] set a point

[4] start training

[5] reset
