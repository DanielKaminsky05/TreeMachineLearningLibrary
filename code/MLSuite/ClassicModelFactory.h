#ifndef CLASSICMODELFACTORY_H
#define CLASSICMODELFACTORY_H

#include "IModel.h"
#include <memory>
#include <string>

class ClassicModelFactory {
public:
    // Creates a Linear Regression model
    std::unique_ptr<IModel> createLinRegModel();

    // Creates a Random Forest model with specified hyperparameters
    std::unique_ptr<IModel> createRandomForestModel(int nEstimators = 100, int maxDepth = 10, int minSamplesSplit = 2);
};

#endif // CLASSICMODELFACTORY_H
