#include "DecisionTree.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

DecisionTree::DecisionTree(int maxDepth, int minSampleSplit)
    : maxDepth(maxDepth),
    minSampleSplit(minSampleSplit),
    nNodes(0),
    nFeatures(0),
    isFitted(false) {}

// Utility: create a new empty node and return its index
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

double DecisionTree::impurityDecrease(int nP, double sumP, double sumP2,
                                      int nL, double sumL, double sumL2,
                                      int nR, double sumR, double sumR2) {
    if (nL == 0 || nR == 0) return 0.0;
    double parentImp = computeMSE(nP, sumP, sumP2);
    double leftImp   = computeMSE(nL, sumL, sumL2);
    double rightImp  = computeMSE(nR, sumR, sumR2);
    // Weighted decrease
    return parentImp - ( (nL * leftImp + nR * rightImp) / nP );
}

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

void DecisionTree::makeLeaf(int nodeIndex,
                            const std::vector<int>& indices,
                            const std::vector<double>& Y) {
    // Mean of Y at this node
    double s = 0.0;
    for (int i : indices) s += Y[i];
    double mean = indices.empty() ? 0.0 : s / indices.size();

    isLeaf[nodeIndex] = true;
    feature[nodeIndex] = -1;
    threshold[nodeIndex] = 0.0;
    left[nodeIndex] = -1;
    right[nodeIndex] = -1;
    value[nodeIndex] = mean;
}

std::tuple<int, double, double, std::vector<int>, std::vector<int>>
DecisionTree::bestSplit(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& Y,
                        const std::vector<int>& indices) {
    // Return: (bestFeat, bestThr, bestGain, bestLeftIdx, bestRightIdx)
    int n = static_cast<int>(indices.size());
    if (n < minSampleSplit || n == 0) {
        return {-1, 0.0, 0.0, {}, {}};
    }

    // Precompute parent sums
    double sumP = 0.0, sumP2 = 0.0;
    for (int i : indices) {
        double y = Y[i];
        sumP += y;
        sumP2 += y * y;
    }

    double bestGain = 0.0;
    int bestFeat = -1;
    double bestThr = 0.0;
    std::vector<int> bestL, bestR;

    for (int f = 0; f < nFeatures; ++f) {
        // Gather (x_f, y, idx) for this subset and sort by feature value
        std::vector<std::tuple<double,double,int>> rows;
        rows.reserve(n);
        for (int i : indices) {
            rows.emplace_back(X[i][f], Y[i], i);
        }
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b){
                      return std::get<0>(a) < std::get<0>(b);
                  });

        // Prefix sums for left, suffix via totals for right
        double sumL = 0.0, sumL2 = 0.0;
        int nL = 0;

        // Sweep possible split points between distinct adjacent feature values
        for (int s = 0; s < n - 1; ++s) {
            double x_s, y_s; int idx_s;
            std::tie(x_s, y_s, idx_s) = rows[s];
            sumL += y_s;
            sumL2 += y_s * y_s;
            ++nL;

            double x_next = std::get<0>(rows[s+1]);
            if (x_s == x_next) {
                // No threshold between equal values—skip
                continue;
            }

            int nR = n - nL;
            if (nL < 1 || nR < 1) continue;

            double sumR  = sumP - sumL;
            double sumR2 = sumP2 - sumL2;

            // Threshold midway between x_s and x_next
            double thr = 0.5 * (x_s + x_next);

            double gain = impurityDecrease(n, sumP, sumP2,
                                           nL, sumL, sumL2,
                                           nR, sumR, sumR2);

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr = thr;

                // Materialize index partitions
                std::vector<int> L; L.reserve(nL);
                std::vector<int> R; R.reserve(nR);
                for (int k = 0; k <= s; ++k) L.push_back(std::get<2>(rows[k]));
                for (int k = s+1; k < n; ++k) R.push_back(std::get<2>(rows[k]));
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
    // Stopping criteria
    if (depth >= maxDepth || static_cast<int>(indices.size()) < minSampleSplit) {
        makeLeaf(nodeIndex, indices, Y);
        return;
    }

    auto [bf, thr, gain, Lidx, Ridx] = bestSplit(X, Y, indices);

    if (bf == -1 || gain <= 0.0) {
        makeLeaf(nodeIndex, indices, Y);
        return;
    }

    // Create children
    int lch = newNode();
    int rch = newNode();

    feature[nodeIndex] = bf;
    threshold[nodeIndex] = thr;
    left[nodeIndex] = lch;
    right[nodeIndex] = rch;
    isLeaf[nodeIndex] = false;
    value[nodeIndex] = 0.0; // unused for internal nodes

    // Recurse
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
