#include "RandomForest.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <limits>
#include <map>

namespace {
	double meanOf(const std::vector<double>& v) {
		if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
		double s = std::accumulate(v.begin(), v.end(), 0.0);
    		return s / static_cast<double>(v.size());
	}

	int clampInt(int x, int lo, int hi) {
    		return std::max(lo, std::min(x, hi));
	}
} 

RandomForest::RandomForest(int Estimators, int maxDepth, int minSamplesSplit, int maxFeatures, bool bootstrap, int randomState, bool isClassification)
    : nEstimators(Estimators),
    maxDepth(maxDepth),
    minSamplesSplit(minSamplesSplit),
    maxFeatures(maxFeatures),
    bootstrap(bootstrap),
    randomState(randomState),
    isClassification(isClassification),
    isFitted(false),
    nFeatures(0),
    internalRng(static_cast<std::mt19937::result_type>(randomState)) // seed RNG
{
	if (nEstimators <= 0) {
        	throw std::invalid_argument("RandomForest: nEstimators must be > 0");
    	}

    	if (minSamplesSplit < 2) {
        	throw std::invalid_argument("RandomForest: minSamplesSplit must be >= 2");
    	}
}

void RandomForest::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
	if (X.empty()) {
        	throw std::invalid_argument("fit: X is empty");
    	}

    	if (X.size() != Y.size()) {
        	throw std::invalid_argument("fit: X and Y size mismatch");
    	}

    	nFeatures = static_cast<int>(X[0].size());

    	if (nFeatures <= 0) {
        	throw std::invalid_argument("fit: X must have at least one feature");
    	}
	
    	// check consistent dims
    	for (const auto& row : X) {
        	if (static_cast<int>(row.size()) != nFeatures) {
            		throw std::invalid_argument("fit: inconsistent feature dimensions in X");
        	}
    	}

    	// Resolve maxFeatures default (0 => floor(sqrt(p)))
    	if (maxFeatures == 0) {
        	maxFeatures = static_cast<int>(std::floor(std::sqrt(static_cast<double>(nFeatures))));
        	maxFeatures = std::max(1, std::min(maxFeatures, nFeatures));
    	} else {
        	maxFeatures = clampInt(maxFeatures, 1, nFeatures);
    	}

    	trees.clear();
    	trees.reserve(static_cast<std::size_t>(nEstimators));

    	for (int t = 0; t < nEstimators; ++t) {
        	buildTree(X, Y);
    	}

    	isFitted = true;
}

void RandomForest::buildTree(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
	const int n = static_cast<int>(X.size());

    	// Choose sample indices (bootstrap or full)
    	std::vector<int> indices;
    	if (bootstrap) {
        	indices = sampleBootstrap(n); // size n with replacement
    	} else {
        	indices.resize(n);
        	std::iota(indices.begin(), indices.end(), 0); // 0..n-1
    	}

    	// materialize the sample
    	std::vector<std::vector<double>> Xb;
    	std::vector<double> Yb;
    	Xb.reserve(indices.size());
    	Yb.reserve(indices.size());
    	for (int idx : indices) {
        	Xb.push_back(X[static_cast<std::size_t>(idx)]);
        	Yb.push_back(Y[static_cast<std::size_t>(idx)]);
    	}

    	DecisionTree tree(maxDepth, minSamplesSplit, isClassification);
    	tree.fit(Xb, Yb);                    

	trees.push_back(std::move(tree));
}

std::vector<int> RandomForest::sampleBootstrap(int n) {
	if (n <= 0) return {};
	std::uniform_int_distribution<int> dist(0, n - 1);
    	std::vector<int> idx;
    	idx.reserve(static_cast<std::size_t>(n));

    	for (int i = 0; i < n; ++i) {
        	idx.push_back(dist(internalRng));
    	}

    	return idx;
}

std::vector<int> RandomForest::sampleFeatures(int p, int k) {
	if (p <= 0 || k <= 0) return {};
    	k = clampInt(k, 1, p);

    	std::vector<int> feats(p);
    	std::iota(feats.begin(), feats.end(), 0);
    	std::shuffle(feats.begin(), feats.end(), internalRng);
    	feats.resize(static_cast<std::size_t>(k));

    	return feats;
}

std::vector<std::vector<double>> RandomForest::predictAllTrees(const std::vector<std::vector<double>>& X) {
	if (!isFitted) {
        	throw std::logic_error("predictAllTrees: model is not fitted");
    	}

    	const std::size_t nSamples = X.size();
    	const std::size_t nTrees   = trees.size();

    	std::vector<std::vector<double>> out(nSamples, std::vector<double>(nTrees, 0.0));
    	for (std::size_t i = 0; i < nSamples; ++i) {
        	for (std::size_t t = 0; t < nTrees; ++t) {
            		out[i][t] = trees[t].predict(X[i]);
        	}
    	}

    	return out;
}

std::vector<double> RandomForest::aggregateMean(const std::vector<double>& preds) {
    return { meanOf(preds) }; // return the mean of predictions 
}

double RandomForest::predict(const std::vector<double>& x) const {
	if (!isFitted) {
        	throw std::logic_error("predict: model is not fitted");
    	}

    	if (static_cast<int>(x.size()) != nFeatures) {
        	throw std::invalid_argument("predict: input dimension does not match training data");
    	}

        if (!isClassification) {
            	// Regression: Mean
            	std::vector<double> perTree;
            	perTree.reserve(trees.size());
            	for (auto& tree : trees) {
                	perTree.push_back(tree.predict(x));
            	}

            	return meanOf(perTree);
        } else {
            	// Classification: Majority Vote Logic
            	std::map<int, int> counts;
            	for (auto& tree : trees) {
                	double p = tree.predict(x);
                	int label = static_cast<int>(std::round(p));
                	counts[label]++;
            	}

            	int bestLabel = -1;
            	int maxCount = -1;

            	for (auto const& [label, count] : counts) {
                	if (count > maxCount) {
                    		maxCount = count;
                    		bestLabel = label;
                	}
            	}
            return static_cast<double>(bestLabel);
        }
}

// model benchmarking interface concrete implementations for the strategy pattern.
std::string RandomForest::getName() const {
	return "Random Forest";
}

void RandomForest::fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) {
	if (x_values.empty() || y_values.empty() || columns.empty()) {
        	throw std::invalid_argument("Input vectors cannot be empty.");
    	}

    	size_t n_cols = columns.size();
    	size_t n_rows = x_values.size() / n_cols;

    	if (x_values.size() % n_cols != 0) {
        	throw std::invalid_argument("The size of x_values is not a multiple of the number of columns.");
    	}

        // since the internal model expects a single scalar target per row, it needs to be decoded
        std::vector<double> targets_double;
        targets_double.reserve(n_rows);

        if (y_values.size() > n_rows) {
		size_t n_target_cols = y_values.size() / n_rows;

		if (y_values.size() % n_rows == 0 && n_target_cols > 1) { // check if it is a clean multiple 
                	for (size_t i = 0; i < n_rows; ++i) {
                    		double maxVal = -std::numeric_limits<double>::infinity();
                    		int maxIdx = 0;

                    		for (size_t k = 0; k < n_target_cols; ++k) {
                        		double val = static_cast<double>(y_values[i * n_target_cols + k]);
                        		if (val > maxVal) {
                            			maxVal = val;
                            			maxIdx = static_cast<int>(k);
                        		}
                    		}
                    		
				targets_double.push_back(static_cast<double>(maxIdx));
                	}
            	} else { // error handling 
                	throw std::invalid_argument("Number of samples in features and targets do not match (and not valid one-hot).");
            	}

        } else if (y_values.size() == n_rows) { // 1-1 scalar mapping for targets 

		for (float v : y_values) {
                	targets_double.push_back(static_cast<double>(v));
            	}

        } else {
             	throw std::invalid_argument("Number of targets is less than number of feature rows.");
        }

    	std::vector<std::vector<double>> features_double(n_rows, std::vector<double>(n_cols)); // reshape then convert to double 
    	for (size_t i = 0; i < n_rows; ++i) {
        	for (size_t j = 0; j < n_cols; ++j) {
            		features_double[i][j] = static_cast<double>(x_values[i * n_cols + j]);
        	}
    	}

    	this->fit(features_double, targets_double);
}

// predict method 
std::vector<float> RandomForest::predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const {
	if (x_values.empty() || columns.empty()) {
		return {};
    	}

    	if (!isFitted) {
        	throw std::logic_error("predict: model is not fitted");
    	}

    	size_t n_cols = columns.size();
    	size_t n_rows = x_values.size() / n_cols;

    	if (x_values.size() % n_cols != 0) {
        	throw std::invalid_argument("The size of x_values is not a multiple of the number of columns.");
    	}

    	std::vector<float> all_predictions;
    	all_predictions.reserve(n_rows);

    	for (size_t i = 0; i < n_rows; ++i) {
        	std::vector<double> single_feature_double; // row of floats to double 
        	single_feature_double.reserve(n_cols);
        	for (size_t j = 0; j < n_cols; ++j) {
            		single_feature_double.push_back(static_cast<double>(x_values[i * n_cols + j]));
        	}

        	if (static_cast<int>(single_feature_double.size()) != nFeatures) {
            		throw std::invalid_argument("predict: input dimension does not match training data");
        	}

            // calls internal predict(std::vector<double>) which handles isClassification check
            all_predictions.push_back(static_cast<float>(this->predict(single_feature_double)));
    }

    return all_predictions;
}
