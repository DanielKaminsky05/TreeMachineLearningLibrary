#include "DecisionTree.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <map>

DecisionTree::DecisionTree(int maxDepth, int minSampleSplit, bool isClassification): 
	maxDepth(maxDepth),
    	minSampleSplit(minSampleSplit),
        isClassification(isClassification),
    	nNodes(0),
    	nFeatures(0),
    	isFitted(false) {}

// create a new empty node and return its index
int DecisionTree::newNode() {
	int id = static_cast<int>(feature.size());
    	feature.push_back(-1);
    	threshold.push_back(0.0);
    	left.push_back(-1);
    	right.push_back(-1);
    	isLeaf.push_back(false);
    	value.push_back(0.0);
    	nNodes = static_cast<int>(feature.size());
    	return id;
}

double DecisionTree::computeMSE(int n, double sum, double sum2) {
    	if (n <= 0) return 0.0;
    	double mean = sum / n;
    	// population MSE of residuals (variance * n / n = variance)
    	return (sum2 / n) - (mean * mean);
}

double DecisionTree::computeGini(const std::vector<int>& indices, const std::vector<double>& Y) {
    if (indices.empty()) return 0.0;
    std::map<double, int> counts;
    for (int idx : indices) {
        counts[Y[idx]]++;
    }
    double n = static_cast<double>(indices.size());
    double sumSq = 0.0;
    for (auto const& [label, count] : counts) {
        double p = count / n;
        sumSq += p * p;
    }
    return 1.0 - sumSq;
}

double DecisionTree::impurityDecrease(int nP, double sumP, double sumP2,
				      int nL, double sumL, double sumL2,
                                      int nR, double sumR, double sumR2,
                                      const std::vector<int>& indicesP,
                                      const std::vector<int>& indicesL,
                                      const std::vector<int>& indicesR,
                                      const std::vector<double>& Y) {
	if (nL == 0 || nR == 0) return 0.0;
    
    if (isClassification) {
        double parentImp = computeGini(indicesP, Y);
        double leftImp   = computeGini(indicesL, Y);
        double rightImp  = computeGini(indicesR, Y);
        return parentImp - ( (static_cast<double>(nL) * leftImp + static_cast<double>(nR) * rightImp) / static_cast<double>(nP) );
    } else {
    	double parentImp = computeMSE(nP, sumP, sumP2);
    	double leftImp   = computeMSE(nL, sumL, sumL2);
    	double rightImp  = computeMSE(nR, sumR, sumR2);
    	return parentImp - ( (static_cast<double>(nL) * leftImp + static_cast<double>(nR) * rightImp) / static_cast<double>(nP) ); // weighted decrease is needed
    }
}

// CART partition 
std::tuple<std::vector<int>, std::vector<int>>
DecisionTree::partitionByThreshold(const std::vector<std::vector<double>>& X,
                                   int feat, double thr,
                                   const std::vector<int>& indices) {
	std::vector<int> L, R;
    	L.reserve(indices.size());
    	R.reserve(indices.size());

    	for (int idx : indices) {
        	if (X[idx][feat] <= thr) L.push_back(idx);
        	else R.push_back(idx);
    	}

    return {L, R};
}

// leaf node definition
void DecisionTree::makeLeaf(int nodeIndex,
                            const std::vector<int>& indices,
                            const std::vector<double>& Y) {
    if (indices.empty()) {
        isLeaf[nodeIndex] = true;
        value[nodeIndex] = 0.0;
        return;
    }

    if (isClassification) {
        // Mode (majority vote)
        std::map<double, int> counts;
        for (int i : indices) counts[Y[i]]++;
        double bestLabel = Y[indices[0]];
        int maxCount = -1;
        for (auto const& [label, count] : counts) {
            if (count > maxCount) {
                maxCount = count;
                bestLabel = label;
            }
        }
        value[nodeIndex] = bestLabel;
    } else {
	    // Mean of Y at this node
	    double s = 0.0;
	    for (int i : indices) s += Y[i];
	    double mean = s / indices.size();
        value[nodeIndex] = mean;
    }

    	isLeaf[nodeIndex] = true;
    	feature[nodeIndex] = -1;
    	threshold[nodeIndex] = 0.0;
    	left[nodeIndex] = -1;
    	right[nodeIndex] = -1;
}

// return a decision tree with the best split params 
// Return: (bestFeat, bestThr, bestGain, bestLeftIdx, bestRightIdx)
std::tuple<int, double, double, std::vector<int>, std::vector<int>>
DecisionTree::bestSplit(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& Y,
                        const std::vector<int>& indices) {

    	int n = static_cast<int>(indices.size());
    	if (n < minSampleSplit || n == 0) {
        	return {-1, 0.0, 0.0, {}, {}};
    	}

    	// parent values
    	double sumP = 0.0, sumP2 = 0.0;
        if (!isClassification) {
    	    for (int i : indices) {
        	    double y = Y[i];
        	    sumP += y;
        	    sumP2 += y * y;
    	    }
        }

    	double bestGain = 0.0;
    	int bestFeat = -1;
    	double bestThr = 0.0;
    	std::vector<int> bestL, bestR;

    	for (int f = 0; f < nFeatures; ++f) {
        	// get (x_f, y, idx) for this subset and sort by feature value
        	std::vector<std::tuple<double,double,int>> rows;
        	rows.reserve(n);
        	for (int i : indices) {
            		rows.emplace_back(X[i][f], Y[i], i);
        	}
        	std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b){
                      return std::get<0>(a) < std::get<0>(b);
                  });

        	// prefix values for left, suffix via totals for right
        	double sumL = 0.0, sumL2 = 0.0;
        	int nL = 0;

        	// sweep all possible split points between distinct adjacent feature values
        	for (int s = 0; s < n - 1; ++s) {
            		double x_s, y_s; int idx_s;
            		std::tie(x_s, y_s, idx_s) = rows[s];
            		sumL += y_s;
            		sumL2 += y_s * y_s;
            		++nL;

            		double x_next = std::get<0>(rows[s+1]);
            		if (x_s == x_next) {
                	// there will be no threshold between equal values—skip
                	continue;
            	}

            	int nR = n - nL;
            	if (nL < 1 || nR < 1) continue;

            	double sumR  = sumP - sumL;
            	double sumR2 = sumP2 - sumL2;

            	// the threshold is midway between x_s and x_next
            	double thr = 0.5 * (x_s + x_next);

                // materialize index partitions for impurityDecrease if needed
                std::vector<int> L; L.reserve(nL); // NOTE: there is a better way of doing this, but simplicity is best for now
                std::vector<int> R; R.reserve(nR);
                for (int k = 0; k <= s; ++k) L.push_back(std::get<2>(rows[k]));
                for (int k = s+1; k < n; ++k) R.push_back(std::get<2>(rows[k]));

            	double gain = impurityDecrease(n, sumP, sumP2, nL, sumL, sumL2, nR, sumR, sumR2, indices, L, R, Y);

            	if (gain > bestGain) {
                	bestGain = gain;
                	bestFeat = f;
                	bestThr = thr;
                    bestL.swap(L);
                    bestR.swap(R);
			    }
        	}
	}

	if (bestFeat == -1) {
        return {-1, 0.0, 0.0, {}, {}};
    }

    return {bestFeat, bestThr, bestGain, bestL, bestR};
}

void DecisionTree::buildTree(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& Y,
                             const std::vector<int>& indices,
                             int depth,
                             int nodeIndex) {
	// stopping criteria
    	if (depth >= maxDepth || static_cast<int>(indices.size()) < minSampleSplit) {
        	makeLeaf(nodeIndex, indices, Y);
        	return;
    	}

    	auto [bf, thr, gain, Lidx, Ridx] = bestSplit(X, Y, indices);

    	if (bf == -1 || gain <= 0.0) {
        	makeLeaf(nodeIndex, indices, Y);
        	return;
    	}

    	// children
    	int lch = newNode();
    	int rch = newNode();

    	feature[nodeIndex] = bf;
    	threshold[nodeIndex] = thr;
    	left[nodeIndex] = lch;
    	right[nodeIndex] = rch;
    	isLeaf[nodeIndex] = false;
    	value[nodeIndex] = 0.0; // not needed for internal nodes

    	// recursive call  
    	buildTree(X, Y, Lidx, depth + 1, lch);
    	buildTree(X, Y, Ridx, depth + 1, rch);
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& Y) {

	if (X.empty() || Y.empty() || X.size() != Y.size()) {
        	throw std::invalid_argument("Fit: X and Y must be non-empty and have the same number of rows.");
    	}

    	nFeatures = static_cast<int>(X[0].size());
    	if (nFeatures == 0) {
        	throw std::invalid_argument("Fit: X must have at least one feature.");
    	}
    	// reset all storage
    	feature.clear(); threshold.clear(); left.clear(); right.clear();
    	isLeaf.clear(); value.clear(); sumY2.clear();
    	nNodes = 0;

    	int root = newNode();
    	std::vector<int> idx(X.size());

    	for (int i = 0; i < (int)X.size(); ++i) idx[i] = i;

    	buildTree(X, Y, idx, /*depth=*/0, root);
    	isFitted = true;
}

double DecisionTree::predict(const std::vector<double>& x) const {
	if (!isFitted) {
        	throw std::runtime_error("predict: model not fitted.");
    	}
	if ((int)x.size() != nFeatures) {
        	throw std::invalid_argument("predict: feature dimension mismatch.");
    	}
	int node = 0; // root
	while (!isLeaf[node]) {
		int f = feature[node];
        	double thr = threshold[node];
        	if (x[f] <= thr) node = left[node];
        	else node = right[node];
        	if (node < 0) break; // safety
    	}
	return value[node < 0 ? 0 : node];
}
