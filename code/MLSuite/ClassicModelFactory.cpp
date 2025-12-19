#include "ClassicModelFactory.h"
// #include "LinRegModel.h"
#include "RandomForestBuilder.h"
#include "LinearRegressionBuilder.h" // Include the builder header
#include "XGBoostBuilder.h"
#include <Eigen/Dense> 
#include <limits>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace {

bool isValidDatasetType(const std::string& type) {
	return type == "train" || type == "test" || type == "val";
}

void ensurePathSet(const std::string& path, const char* label) {
	if (path.empty()) {
		throw std::invalid_argument(std::string("ClassicModelFactory: missing ") + label + " path.");
	}
}

void ensureTypeValid(const std::string& type, const char* label) {
	if (!isValidDatasetType(type)) {
		throw std::invalid_argument(std::string("ClassicModelFactory: invalid ") + label + " type: " + type);
	}
}

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

ClassicModelFactory::ClassicModelFactory(const std::string& trainFeaturesPath,
	const std::string& trainTargetsPath,
	const std::string& testFeaturesPath,
	const std::string& testTargetsPath,
	const std::string& trainType,
	const std::string& testType)
	: m_trainFeaturesPath(trainFeaturesPath),
	m_trainTargetsPath(trainTargetsPath),
	m_testFeaturesPath(testFeaturesPath),
	m_testTargetsPath(testTargetsPath),
	m_trainType(trainType),
	m_testType(testType) {}

void ClassicModelFactory::setTrainDataPaths(const std::string& featuresPath,
	const std::string& targetsPath,
	const std::string& type) {
	ensureTypeValid(type, "train");
	m_trainFeaturesPath = featuresPath;
	m_trainTargetsPath = targetsPath;
	m_trainType = type;
}

void ClassicModelFactory::setTestDataPaths(const std::string& featuresPath,
	const std::string& targetsPath,
	const std::string& type) {
	ensureTypeValid(type, "test");
	m_testFeaturesPath = featuresPath;
	m_testTargetsPath = targetsPath;
	m_testType = type;
}

const std::string& ClassicModelFactory::getTrainFeaturesPath() const {
	return m_trainFeaturesPath;
}

const std::string& ClassicModelFactory::getTrainTargetsPath() const {
	return m_trainTargetsPath;
}

const std::string& ClassicModelFactory::getTrainType() const {
	return m_trainType;
}

const std::string& ClassicModelFactory::getTestFeaturesPath() const {
	return m_testFeaturesPath;
}

const std::string& ClassicModelFactory::getTestTargetsPath() const {
	return m_testTargetsPath;
}

const std::string& ClassicModelFactory::getTestType() const {
	return m_testType;
}

Dataset ClassicModelFactory::loadTrainFeatures() const {
	ensurePathSet(m_trainFeaturesPath, "train features");
	ensureTypeValid(m_trainType, "train");
	return Dataset(m_trainFeaturesPath, m_trainType);
}

Dataset ClassicModelFactory::loadTrainTargets() const {
	ensurePathSet(m_trainTargetsPath, "train targets");
	ensureTypeValid(m_trainType, "train");
	return Dataset(m_trainTargetsPath, m_trainType);
}

Dataset ClassicModelFactory::loadTestFeatures() const {
	ensurePathSet(m_testFeaturesPath, "test features");
	ensureTypeValid(m_testType, "test");
	return Dataset(m_testFeaturesPath, m_testType);
}

Dataset ClassicModelFactory::loadTestTargets() const {
	ensurePathSet(m_testTargetsPath, "test targets");
	ensureTypeValid(m_testType, "test");
	return Dataset(m_testTargetsPath, m_testType);
}

void ClassicModelFactory::fitModel(IModel& model) const {
	Dataset features = loadTrainFeatures();
	Dataset targets = loadTrainTargets();
	model.fit(features.get_data(), features.get_columns(), targets.get_data());
}

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
    const int kFolds = 5;

    // Prepare shuffled indices for K-Fold Cross Validation
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

	double bestAvgMSE = std::numeric_limits<double>::infinity();
	std::unique_ptr<IModel> bestModel;

    // Parameters to store the best configuration found
    // RandomForest
    int bestRF_nEstimators = 0;
    int bestRF_maxDepth = 0;
    int bestRF_minSamplesSplit = 0;

    // XGBoost
    int bestXGB_nEstimators = 0;
    float bestXGB_learningRate = 0.0f;
    int bestXGB_maxDepth = 0;
    float bestXGB_subsampleRatio = 0.0f;
    float bestXGB_gamma = 0.0f;
    std::string bestXGB_regularization;

    bool foundAny = false;

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

            double totalMSE = 0.0;

            // K-Fold Loop
            for (int k = 0; k < kFolds; ++k) {
                std::vector<std::vector<double>> trainX, valX;
                std::vector<double> trainY, valY;
                
                // Reserve memory to avoid reallocations
                trainX.reserve(X.size()); 
                valX.reserve(X.size() / kFolds + 2);
                trainY.reserve(y.size()); 
                valY.reserve(y.size() / kFolds + 2);

                size_t foldSize = X.size() / kFolds;
                size_t start = k * foldSize;
                size_t end = (k == kFolds - 1) ? X.size() : start + foldSize;

                for (size_t i = 0; i < X.size(); ++i) {
                    if (i >= start && i < end) {
                        valX.push_back(X[indices[i]]);
                        valY.push_back(y[indices[i]]);
                    } else {
                        trainX.push_back(X[indices[i]]);
                        trainY.push_back(y[indices[i]]);
                    }
                }

                auto rf = RandomForestBuilder()
                    .setEstimators(nEstimators)
                    .setMaxDepth(maxDepth)
                    .setMinSamplesSplit(minSamplesSplit)
                    .build();

                rf->fit(trainX, trainY);
                totalMSE += computeMSE(*rf, valX, valY);
            }

            double avgMSE = totalMSE / kFolds;

			if (avgMSE < bestAvgMSE) {
				bestAvgMSE = avgMSE;
                bestRF_nEstimators = nEstimators;
                bestRF_maxDepth = maxDepth;
                bestRF_minSamplesSplit = minSamplesSplit;
                foundAny = true;
			}
		}

        // Rebuild best model on full dataset
        if (foundAny) {
            auto finalRf = RandomForestBuilder()
                .setEstimators(bestRF_nEstimators)
                .setMaxDepth(bestRF_maxDepth)
                .setMinSamplesSplit(bestRF_minSamplesSplit)
                .build();
            finalRf->fit(X, y);
            bestModel = std::move(finalRf);
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

            double totalMSE = 0.0;

            // K-Fold Loop
            for (int k = 0; k < kFolds; ++k) {
                std::vector<std::vector<double>> trainX, valX;
                std::vector<double> trainY, valY;

                trainX.reserve(X.size()); 
                valX.reserve(X.size() / kFolds + 2);
                trainY.reserve(y.size()); 
                valY.reserve(y.size() / kFolds + 2);

                size_t foldSize = X.size() / kFolds;
                size_t start = k * foldSize;
                size_t end = (k == kFolds - 1) ? X.size() : start + foldSize;

                for (size_t i = 0; i < X.size(); ++i) {
                    if (i >= start && i < end) {
                        valX.push_back(X[indices[i]]);
                        valY.push_back(y[indices[i]]);
                    } else {
                        trainX.push_back(X[indices[i]]);
                        trainY.push_back(y[indices[i]]);
                    }
                }

                auto xgb = XGBoostBuilder()
                    .setNEstimators(nEstimators)
                    .setLearningRate(learningRate)
                    .setMaxDepth(maxDepth)
                    .setSubsampleRatio(subsampleRatio)
                    .setGamma(gamma)
                    .setRegularization(regularization)
                    .build();

                xgb->fit(trainX, trainY);
                totalMSE += computeMSE(*xgb, valX, valY);
            }

            double avgMSE = totalMSE / kFolds;

			if (avgMSE < bestAvgMSE) {
				bestAvgMSE = avgMSE;
                bestXGB_nEstimators = nEstimators;
                bestXGB_learningRate = learningRate;
                bestXGB_maxDepth = maxDepth;
                bestXGB_subsampleRatio = subsampleRatio;
                bestXGB_gamma = gamma;
                bestXGB_regularization = regularization;
                foundAny = true;
			}
		}

        // Rebuild best model on full dataset
        if (foundAny) {
            auto finalXgb = XGBoostBuilder()
                .setNEstimators(bestXGB_nEstimators)
                .setLearningRate(bestXGB_learningRate)
                .setMaxDepth(bestXGB_maxDepth)
                .setSubsampleRatio(bestXGB_subsampleRatio)
                .setGamma(bestXGB_gamma)
                .setRegularization(bestXGB_regularization)
                .build();
            finalXgb->fit(X, y);
            bestModel = std::move(finalXgb);
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