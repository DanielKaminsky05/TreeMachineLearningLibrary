#include "ClassicModelFactory.h"
// #include "LinRegModel.h"
#include "RandomForestBuilder.h"
#include "LinearRegressionBuilder.h" // Include the builder header
#include <Eigen/Dense> // Keep if still needed by other parts of the factory

std::unique_ptr<IModel> ClassicModelFactory::createLinRegModel() {
    // Use the builder to create an unfitted LinRegModel
    return LinearRegressionBuilder().build_unfitted();
}

// you cannot return a random forest that is unfitted, so it will build and fit the model in one go.
std::unique_ptr<IModel> ClassicModelFactory::createRandomForestModel(int nEstimators, int maxDepth, int minSamplesSplit) {
    return RandomForestBuilder()
        .setEstimators(nEstimators)
        .setMaxDepth(maxDepth)
        .setMinSamplesSplit(minSamplesSplit)
        .setMaxFeatures(0) 
        .setBootstrap(true)
	.setRandomState(0)
        .build();
}

