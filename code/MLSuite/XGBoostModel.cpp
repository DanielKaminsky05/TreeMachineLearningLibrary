#include "XGBoostModel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

// ---------- Helpers ----------
double XGBoostModel::sigmoid(double z) {
    // Numerically stable-ish sigmoid
    if (z >= 0) {
        double e = std::exp(-z);
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(z);
        return e / (1.0 + e);
    }
}

double XGBoostModel::logit(double p) {
    // clip to avoid infinities
    double pc = clipped(p, 1e-12, 1.0 - 1e-12);
    return std::log(pc / (1.0 - pc));
}

double XGBoostModel::clipped(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// ---------- Constructor ----------
XGBoostModel::XGBoostModel(int n_est_counts, float learning_rate, std::string loss_fn,
                           int depth, float subsample_ratio, float gamma, std::string regularization)
    : n_est_counts(n_est_counts),
    learning_rate(learning_rate),
    loss_fn(std::move(loss_fn)),
    depth(depth),
    subsample_ratio(subsample_ratio),
    gamma(gamma),
    regularization(std::move(regularization)) {
    if (this->subsample_ratio <= 0.0f || this->subsample_ratio > 1.0f) {
        throw std::invalid_argument("subsample_ratio must be in (0, 1].");
    }
    if (this->n_est_counts <= 0) {
        throw std::invalid_argument("n_est_counts must be > 0.");
    }
    if (this->depth <= 0) {
        throw std::invalid_argument("depth must be > 0.");
    }
    if (this->learning_rate <= 0.0f) {
        throw std::invalid_argument("learning_rate must be > 0.");
    }
}

// ---------- fit ----------
void XGBoostModel::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& Y) {
    const size_t n = Y.size();
    if (n == 0 || X.size() == 0 || X.size() != n) {
        throw std::invalid_argument("X and Y must be non-empty and have the same number of rows.");
    }
    if (X[0].size() == 0) {
        throw std::invalid_argument("X must have at least one feature.");
    }

    trees.clear();
    trees.reserve(static_cast<size_t>(n_est_counts));

    // Initialize predictions depending on loss
    std::vector<double> y_pred(n, 0.0);

    if (loss_fn == "binary:logistic") {
        // Bias is logit of positive rate
        double pos_rate = std::accumulate(Y.begin(), Y.end(), 0.0) / static_cast<double>(n);
        init_bias = logit(pos_rate);
        std::fill(y_pred.begin(), y_pred.end(), init_bias); // raw scores (log-odds)
    } else {
        // Default: reg:squarederror
        double mean = std::accumulate(Y.begin(), Y.end(), 0.0) / static_cast<double>(n);
        init_bias = mean;
        std::fill(y_pred.begin(), y_pred.end(), init_bias);
    }

    // PRNG for subsampling
    std::mt19937 rng(42); // fixed seed for reproducibility; you can expose/set this

    for (int m = 0; m < n_est_counts; ++m) {
        // Compute pseudo-residuals
        std::vector<double> residuals(n);

        if (loss_fn == "binary:logistic") {
            // Gradient of logistic loss wrt raw score f(x): grad = sigmoid(f) - y
            for (size_t i = 0; i < n; ++i) {
                double p = sigmoid(y_pred[i]);
                double grad = (p - Y[i]);     // grad
                residuals[i] = -grad;         // fit tree to negative gradient
            }
        } else {
            // reg:squarederror: grad = (y_pred - y); residual = y - y_pred
            for (size_t i = 0; i < n; ++i) {
                residuals[i] = (Y[i] - y_pred[i]);
            }
        }

        // Row subsampling (without replacement)
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        size_t k = static_cast<size_t>(std::ceil(subsample_ratio * static_cast<float>(n)));
        k = std::max<size_t>(1, std::min(k, n));

        std::vector<std::vector<double>> X_sub;
        std::vector<double> r_sub;
        X_sub.reserve(k);
        r_sub.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            size_t j = idx[i];
            X_sub.push_back(X[j]);
            r_sub.push_back(residuals[j]);
        }

        // Train a shallow tree on residuals
        DecisionTree tree(depth /*maxDepth*/);
        tree.Fit(X_sub, r_sub);

        // Save it
        trees.push_back(std::move(tree));

        // Update predictions on full data
        for (size_t i = 0; i < n; ++i) {
            double tpred = trees.back().predict(X[i]);
            y_pred[i] += static_cast<double>(learning_rate) * tpred;
        }
    }

    is_fitted = true;
}

// ---------- predict ----------
double XGBoostModel::predict(const std::vector<double> input) {
    if (!is_fitted) {
        throw std::runtime_error("Model not fitted. Call fit() first.");
    }

    double score = init_bias;
    for (const auto& t : trees) {
        score += static_cast<double>(learning_rate) * t.predict(input);
    }

    if (loss_fn == "binary:logistic") {
        // Return probability by default
        return sigmoid(score);
    }
    // Regression: return raw prediction
    return score;
}
