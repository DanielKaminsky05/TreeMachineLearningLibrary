#include "ClassicModelFactory.h"
// #include "LinRegModel.h"
#include "RandomForestBuilder.h"
#include "LinearRegressionBuilder.h" // Include the builder header
#include "XGBoostBuilder.h"
#include <Eigen/Dense> // Keep if still needed by other parts of the fact#include <limits>
#include <random>
#include <stdexcept>

namespace {

// Helper to compute Mean Squared Error (MSE) for models that expose:
//   double predict(const std::vector<double>&)
template <typename ModelT>
double computeMSE(ModelT& model,
                  const std::vector<std::vector<double>>& X,
                  const std::vector<double>& y) {
	if (X.empty() || y.empty() || X.size() != y.size()) {
		throw std::invalid_argument("computeMSE: X and y must be non-empty and have matching sizes.");
	}

	double sumSq = 0.0;
	for (std::size_t i = 0; i < X.size(); ++i) {
		double pred = model.predict(X[i]);
		double diff = pred - y[i];
		sumSq += diff * diff;
	}
	return sumSq / static_cast<double>(X.size());
}

// Utility to pick a random element from a vector of strings
const std::string& pickRandom(const std::vector<std::string>& values, std::mt19937& rng) {
	if (values.empty()) {
		throw std::invalid_argument("pickRandom: hyperparameter value list cannot be empty.");
	}
	std::uniform_int_distribution<std::size_t> dist(0, values.size() - 1);
	return values[dist(rng)];
}

} // namespace

std::unique_ptr<IModel> ClassicModelFactory::randomSearch(
	const std::string& modelType,
        const std::vector<std::vector<std::string>>& hyperParams,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y) {

	if (X.empty() || y.empty() || X.size() != y.size()) {
		throw std::invalid_argument("randomSearch: X and y must be non-empty and have matching sizes.");
	}

	if (modelType != "RandomForest" && modelType != "XGBoost") {
		throw std::invalid_argument("randomSearch currently supports only \"RandomForest\" and \"XGBoost\" model types.");
	}

	if (hyperParams.empty()) {
		throw std::invalid_argument("randomSearch: hyperParams cannot be empty.");
	}

	// Fixed seed for reproducibility
	std::mt19937 rng(42u);
	const int maxIterations = 20;

	double bestMSE = std::numeric_limits<double>::infinity();
	std::unique_ptr<IModel> bestModel;

	if (modelType == "RandomForest") {
		// Expected order:
		//   hyperParams[0] -> candidates for nEstimators (int)
		//   hyperParams[1] -> candidates for maxDepth (int)
		//   hyperParams[2] -> candidates for minSamplesSplit (int)
		if (hyperParams.size() < 3) {
			throw std::invalid_argument(
				"randomSearch(RandomForest): expected at least 3 hyperparameter lists: "
				"nEstimators, maxDepth, minSamplesSplit.");
		}

		const auto& nEstimatorsVals      = hyperParams[0];
		const auto& maxDepthVals         = hyperParams[1];
		const auto& minSamplesSplitVals  = hyperParams[2];

		for (int iter = 0; iter < maxIterations; ++iter) {
			int nEstimators     = std::stoi(pickRandom(nEstimatorsVals, rng));
			int maxDepth        = std::stoi(pickRandom(maxDepthVals, rng));
			int minSamplesSplit = std::stoi(pickRandom(minSamplesSplitVals, rng));

			auto rf = RandomForestBuilder()
				.setEstimators(nEstimators)
				.setMaxDepth(maxDepth)
				.setMinSamplesSplit(minSamplesSplit)
				.build();

			rf->fit(X, y);
			double mse = computeMSE(*rf, X, y);

			if (mse < bestMSE) {
				bestMSE = mse;
				bestModel = std::move(rf);
			}
		}
	} else if (modelType == "XGBoost") {
		// Expected order:
		//   hyperParams[0] -> candidates for nEstimators (int)
		//   hyperParams[1] -> candidates for learningRate (float)
		//   hyperParams[2] -> candidates for maxDepth (int)
		//   hyperParams[3] -> candidates for subsampleRatio (float)
		//   hyperParams[4] -> candidates for gamma (float)
		//   hyperParams[5] -> candidates for regularization (string)
		if (hyperParams.size() < 6) {
			throw std::invalid_argument(
				"randomSearch(XGBoost): expected at least 6 hyperparameter lists: "
				"nEstimators, learningRate, maxDepth, subsampleRatio, gamma, regularization.");
		}

		const auto& nEstimatorsVals    = hyperParams[0];
		const auto& learningRateVals   = hyperParams[1];
		const auto& maxDepthVals       = hyperParams[2];
		const auto& subsampleVals      = hyperParams[3];
		const auto& gammaVals          = hyperParams[4];
		const auto& regularizationVals = hyperParams[5];

		for (int iter = 0; iter < maxIterations; ++iter) {
			int nEstimators        = std::stoi(pickRandom(nEstimatorsVals, rng));
			float learningRate     = std::stof(pickRandom(learningRateVals, rng));
			int maxDepth           = std::stoi(pickRandom(maxDepthVals, rng));
			float subsampleRatio   = std::stof(pickRandom(subsampleVals, rng));
			float gamma            = std::stof(pickRandom(gammaVals, rng));
			std::string regularization = pickRandom(regularizationVals, rng);

			auto xgb = XGBoostBuilder()
				.setNEstimators(nEstimators)
				.setLearningRate(learningRate)
				.setMaxDepth(maxDepth)
				.setSubsampleRatio(subsampleRatio)
				.setGamma(gamma)
				.setRegularization(regularization)
				.build();

			xgb->fit(X, y);
			double mse = computeMSE(*xgb, X, y);

			if (mse < bestMSE) {
				bestMSE = mse;
				bestModel = std::move(xgb);
			}
		}
	}

	return bestModel;
}

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

std::unique_ptr<IModel> ClassicModelFactory::createXGBoostModel(int nEstimators, float learningRate, int maxDepth, float subsampleRatio, float gamma, const std::string& regularization) {
    	return XGBoostBuilder()
        	.setNEstimators(nEstimators)
        	.setLearningRate(learningRate)
        	.setMaxDepth(maxDepth)
        	.setSubsampleRatio(subsampleRatio)
        	.setGamma(gamma)
        	.setRegularization(regularization)
        	.build();
}



