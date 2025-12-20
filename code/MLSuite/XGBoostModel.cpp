#include "XGBoostModel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
    // sigmoid function for binary classification
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
}

XGBoostModel::XGBoostModel(int nEstimatorsValue,
                           float learningRateValue,
                           int maxDepthValue,
                           float subsampleRatioValue,
                           float gammaValue,
                           std::string regularizationValue,
                           bool isClassificationValue)
    : nEstimators(nEstimatorsValue),
      learningRate(learningRateValue),
      maxDepth(maxDepthValue),
      subsampleRatio(subsampleRatioValue),
      gamma(gammaValue),
      regularization(std::move(regularizationValue)),
      isClassification(isClassificationValue) {
	
	if (subsampleRatio <= 0.0f || subsampleRatio > 1.0f) {
        	throw std::invalid_argument("subsampleRatio must be in (0, 1].");
    	}

    	if (nEstimators <= 0) {
        	throw std::invalid_argument("nEstimators must be > 0.");
    	}

    	if (maxDepth <= 0) {
        	throw std::invalid_argument("maxDepth must be > 0.");
    	}

    	if (learningRate <= 0.0f) {
        	throw std::invalid_argument("learningRate must be > 0.");
    	}
}

void XGBoostModel::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    	const size_t sampleCount = Y.size();
    	if (sampleCount == 0 || X.empty() || X.size() != sampleCount) {
        	throw std::invalid_argument("X and Y must be non-empty and have matching rows.");
    	}

    	if (X[0].empty()) {
        	throw std::invalid_argument("X must contain at least one feature.");
    	}

    	trees.clear();
    	trees.reserve(static_cast<size_t>(nEstimators));

        if (isClassification) {
            // using 0.0 for simplicity or log-odds of mean.
            double posCount = 0.0;
            for(double y : Y) if(y > 0.5) posCount++;
            double prob = posCount / sampleCount;
            
            // prevent log(0)
            prob = std::max(1e-6, std::min(1.0 - 1e-6, prob));
            initialBias = std::log(prob / (1.0 - prob));
        } else {
    	    double meanTarget = std::accumulate(Y.begin(), Y.end(), 0.0) / static_cast<double>(sampleCount);
    	    initialBias = meanTarget;
        }

    	std::vector<double> predictions(sampleCount, initialBias);
    	std::vector<double> residuals(sampleCount);
	
	    std::mt19937 rng(42);

    	for (int treeIndex = 0; treeIndex < nEstimators; ++treeIndex) {
        	for (size_t i = 0; i < sampleCount; ++i) {
                if (isClassification) {
                    double prob = sigmoid(predictions[i]); // fit the tree to the gradients using log loss 
                    residuals[i] = Y[i] - prob; 
                } else {
            		residuals[i] = Y[i] - predictions[i]; // MSE: gradient = y - pred
                }
        	}

        	std::vector<size_t> indices(sampleCount);
        	std::iota(indices.begin(), indices.end(), 0);
        	std::shuffle(indices.begin(), indices.end(), rng);

        	size_t subsampleSize = static_cast<size_t>(std::ceil(subsampleRatio * static_cast<float>(sampleCount)));
        	subsampleSize = std::max<size_t>(1, std::min(subsampleSize, sampleCount));

        	std::vector<std::vector<double>> featureSubset;
        	std::vector<double> residualSubset;
        	featureSubset.reserve(subsampleSize);
        	residualSubset.reserve(subsampleSize);

        	for (size_t i = 0; i < subsampleSize; ++i) {
            		size_t rowIndex = indices[i];
            		featureSubset.push_back(X[rowIndex]);
            		residualSubset.push_back(residuals[rowIndex]);
        	}

            DecisionTree tree(maxDepth, 2, false); 
        	tree.fit(featureSubset, residualSubset);
        	trees.push_back(std::move(tree));

        	for (size_t i = 0; i < sampleCount; ++i) {
            		double treePrediction = trees.back().predict(X[i]);
            		predictions[i] += static_cast<double>(learningRate) * treePrediction;
        	}
    	}

    	isFitted = true;
}

double XGBoostModel::predict(const std::vector<double>& input) const {
    	if (!isFitted) {
        	throw std::runtime_error("Model not fitted. Call fit() first.");
    	}

    	double score = initialBias;
    	for (const auto& tree : trees) {
        	score += static_cast<double>(learningRate) * tree.predict(input);
    	}

        if (isClassification) { // return binary 1 or 0 depending on probability 
            double prob = sigmoid(score);
            return (prob >= 0.5) ? 1.0 : 0.0;
        }

    	return score;
}

void XGBoostModel::fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) {
	if (columns.empty()) {
        	throw std::invalid_argument("Columns must be provided for XGBoostModel::fit.");
    	}

    	if (x_values.empty() || y_values.empty()) {
        	throw std::invalid_argument("Feature and target vectors must be non-empty.");
    	}

    	const size_t columnCount = columns.size();

    	if (x_values.size() % columnCount != 0) {
        	throw std::invalid_argument("Feature vector size must be a multiple of the number of columns.");
    	}

    	const size_t rowCount = x_values.size() / columnCount;

        // Check for 1:1 mapping first
        if (y_values.size() == rowCount) {
             std::vector<std::vector<double>> features(rowCount, std::vector<double>(columnCount));
    	     for (size_t i = 0; i < rowCount; ++i) {
        	    for (size_t j = 0; j < columnCount; ++j) {
            		features[i][j] = static_cast<double>(x_values[i * columnCount + j]);
        	    }
    	     }
    	     std::vector<double> targets(y_values.begin(), y_values.end());
    	     fit(features, targets);
             return;
        }

    	if (rowCount != y_values.size()) { // check for encoding mismatch 
        	throw std::invalid_argument("Feature rows must match target size (XGBoost only supports single-output regression/binary classification).");
    	}
}

std::vector<float> XGBoostModel::predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const {
	if (!isFitted) {
		throw std::runtime_error("Model not fitted. Call fit() before predict().");
    	}

    	if (x_values.empty() || columns.empty()) {
        	return {};
    	}

    	const size_t columnCount = columns.size();

    	if (x_values.size() % columnCount != 0) {
        	throw std::invalid_argument("Feature vector size must be a multiple of the number of columns.");
    	}

    	const size_t rowCount = x_values.size() / columnCount;
    	std::vector<float> predictions;
    	predictions.reserve(rowCount);

    	for (size_t i = 0; i < rowCount; ++i) {
        	std::vector<double> sample(columnCount);

        	for (size_t j = 0; j < columnCount; ++j) {
            		sample[j] = static_cast<double>(x_values[i * columnCount + j]);
        	}
        	predictions.push_back(static_cast<float>(predict(sample)));
    	}

	return predictions;
}
