#include "ClassicModelFactory.h"
#include "LinRegModel.h"
#include "RandomForest.h"
#include "LinearRegressionBuilder.h" // Include the builder header
#include <Eigen/Dense> // Keep if still needed by other parts of the factory

std::unique_ptr<IModel> ClassicModelFactory::createLinRegModel() {
    // Use the builder to create an unfitted LinRegModel
    return LinearRegressionBuilder().build_unfitted();
}

std::unique_ptr<IModel> ClassicModelFactory::createRandomForestModel(int nEstimators, int maxDepth, int minSamplesSplit) {
    // Using default values for the last 3 parameters of the RandomForest constructor:
    // (maxFeatures=0, bootstrap=true, randomState=0)
    return std::make_unique<RandomForest>(nEstimators, maxDepth, minSamplesSplit, 0, true, 0);
}
