#ifndef CLASSICMODELFACTORY_H
#define CLASSICMODELFACTORY_H

#include "IModel.h"
#include <memory>
#include <string>

class ClassicModelFactory {
public:
	std::unique_ptr<IModel> createLinRegModel(); // linreg 

	std::unique_ptr<IModel> createRandomForestModel(int nEstimators = 100, int maxDepth = 10, int minSamplesSplit = 2); // random forest

	std::unique_ptr<IModel> createXGBoostModel(int nEstimators = 100, float learningRate = 0.1f, int maxDepth = 3, float subsampleRatio = 1.0f, float gamma = 0.0f, const std::string& regularization = "L2"); // XGBoost 



};

#endif 
