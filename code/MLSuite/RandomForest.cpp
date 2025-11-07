#include "RandomForest.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <limits>

namespace {
double meanOf(const std::vector<double>& v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / static_cast<double>(v.size());
}

int clampInt(int x, int lo, int hi) {
    return std::max(lo, std::min(x, hi));
}
} // namespace

RandomForest::RandomForest(int Estimators,
                           int maxDepth,
                           int minSamplesSplit,
                           int maxFeatures,
                           bool bootstrap,
                           int randomState)
    : nEstimators(Estimators),
    maxDepth(maxDepth),
    minSamplesSplit(minSamplesSplit),
    maxFeatures(maxFeatures),
    bootstrap(bootstrap),
    randomState(randomState),
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

void RandomForest::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& Y)
{
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

void RandomForest::buildTree(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& Y)
{
    const int n = static_cast<int>(X.size());

    // Choose sample indices (bootstrap or full)
    std::vector<int> indices;
    if (bootstrap) {
        indices = sampleBootstrap(n);                // size n with replacement
    } else {
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0); // 0..n-1
    }

    // Materialize the sample
    std::vector<std::vector<double>> Xb;
    std::vector<double> Yb;
    Xb.reserve(indices.size());
    Yb.reserve(indices.size());
    for (int idx : indices) {
        Xb.push_back(X[static_cast<std::size_t>(idx)]);
        Yb.push_back(Y[static_cast<std::size_t>(idx)]);
    }

    // (Optional) if your DecisionTree later supports randomness, you can draw a seed:
    // unsigned int seed = internalRng();

    // Construct a DecisionTree with your API and train it
    DecisionTree tree(maxDepth, minSamplesSplit);
    tree.Fit(Xb, Yb);                    // <-- matches your DecisionTree

    trees.push_back(std::move(tree));
}

std::vector<int> RandomForest::sampleBootstrap(int n)
{
    if (n <= 0) return {};
    std::uniform_int_distribution<int> dist(0, n - 1);
    std::vector<int> idx;
    idx.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx.push_back(dist(internalRng));
    }
    return idx;
}

std::vector<int> RandomForest::sampleFeatures(int p, int k)
{
    if (p <= 0 || k <= 0) return {};
    k = clampInt(k, 1, p);

    std::vector<int> feats(p);
    std::iota(feats.begin(), feats.end(), 0);
    std::shuffle(feats.begin(), feats.end(), internalRng);
    feats.resize(static_cast<std::size_t>(k));
    return feats;
}

std::vector<std::vector<double>> RandomForest::predictAllTrees(
    const std::vector<std::vector<double>>& X)
{
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

std::vector<double> RandomForest::aggregateMean(const std::vector<double>& preds)
{
    // Your header returns vector<double>; we return {mean} to match it.
    return { meanOf(preds) };
}

double RandomForest::predict(const std::vector<double>& x)
{
    if (!isFitted) {
        throw std::logic_error("predict: model is not fitted");
    }
    if (static_cast<int>(x.size()) != nFeatures) {
        throw std::invalid_argument("predict: input dimension does not match training data");
    }

    std::vector<double> perTree;
    perTree.reserve(trees.size());
    for (auto& tree : trees) {
        perTree.push_back(tree.predict(x));
    }
    return meanOf(perTree);
}
