// BenchmarkStrategy.cpp
#include "BenchmarkStrategy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

#include "DecisionTree.h" // for the DecisionTree convenience wrapper

namespace mlsuite {

using clock_type = std::chrono::steady_clock;
using seconds_d  = std::chrono::duration<double>;

std::string BenchmarkResult::toString() const {
    std::string s;
    s += "Train samples: " + std::to_string(n_train_samples) + 
         ", features: " + std::to_string(n_train_features) + "\n";
    s += "Test  samples: " + std::to_string(n_test_samples) + 
         ", features: " + std::to_string(n_test_features) + "\n";
    s += "Train time (s):  " + std::to_string(train_seconds) + "\n";
    s += "Predict time (s):" + std::to_string(predict_seconds) + "\n";
    s += "MSE:  " + std::to_string(mse)  + "\n";
    s += "RMSE: " + std::to_string(rmse) + "\n";
    s += "MAE:  " + std::to_string(mae)  + "\n";
    s += "R2:   " + std::to_string(r2)   + "\n";
    if (peak_memory_kb >= 0) {
        s += "Peak memory (KB): " + std::to_string(peak_memory_kb) + "\n";
    }
    return s;
}

// static
double RegressionBenchmarkStrategy::calc_mse(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) {
    const size_t n = std::min(y_true.size(), y_pred.size());
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double e = y_true[i] - y_pred[i];
        sum += e * e;
    }
    return sum / static_cast<double>(n);
}

// static
double RegressionBenchmarkStrategy::calc_mae(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) {
    const size_t n = std::min(y_true.size(), y_pred.size());
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::abs(y_true[i] - y_pred[i]);
    }
    return sum / static_cast<double>(n);
}

// static
double RegressionBenchmarkStrategy::calc_r2(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) {
    const size_t n = std::min(y_true.size(), y_pred.size());
    if (n == 0) return 0.0;
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += y_true[i];
    mean /= static_cast<double>(n);

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = y_true[i] - y_pred[i];
        ss_res += diff * diff;
        const double dev = y_true[i] - mean;
        ss_tot += dev * dev;
    }
    if (ss_tot == 0.0) return 0.0; // undefined; return 0 to avoid NaN
    return 1.0 - (ss_res / ss_tot);
}

BenchmarkResult RegressionBenchmarkStrategy::run_with_functions(
    const std::function<void()>& train_fn,
    const std::function<std::vector<double>()>& predict_fn,
    const std::vector<double>& y_true,
    size_t n_train_samples,
    size_t n_train_features,
    size_t n_test_samples,
    size_t n_test_features
) const {
    BenchmarkResult res{};
    res.n_train_samples  = n_train_samples;
    res.n_train_features = n_train_features;
    res.n_test_samples   = n_test_samples;
    res.n_test_features  = n_test_features;

    // Train timing
    const auto t0 = clock_type::now();
    train_fn();
    const auto t1 = clock_type::now();
    res.train_seconds = std::chrono::duration_cast<seconds_d>(t1 - t0).count();

    // Predict timing
    const auto p0 = clock_type::now();
    const std::vector<double> y_pred = predict_fn();
    const auto p1 = clock_type::now();
    res.predict_seconds = std::chrono::duration_cast<seconds_d>(p1 - p0).count();

    // Metrics
    res.mse  = calc_mse(y_true, y_pred);
    res.rmse = std::sqrt(res.mse);
    res.mae  = calc_mae(y_true, y_pred);
    res.r2   = calc_r2(y_true, y_pred);

    return res;
}

BenchmarkResult RegressionBenchmarkStrategy::run_decision_tree(
    DecisionTree& model,
    const std::vector<std::vector<double>>& X_train,
    const std::vector<double>& y_train,
    const std::vector<std::vector<double>>& X_test,
    const std::vector<double>& y_test
) const {
    const size_t ntr = X_train.size();
    const size_t dtr = ntr ? X_train[0].size() : 0;
    const size_t nte = X_test.size();
    const size_t dte = nte ? X_test[0].size() : 0;

    auto train_fn = [&]() {
        model.Fit(X_train, y_train);
    };

    auto predict_fn = [&]() -> std::vector<double> {
        std::vector<double> out;
        out.reserve(X_test.size());
        for (const auto& x : X_test) {
            out.push_back(model.predict(x));
        }
        return out;
    };

    return run_with_functions(train_fn, predict_fn, y_test, ntr, dtr, nte, dte);
}

std::string format_result_pretty(const BenchmarkResult& r, const std::string& model_name) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(6);
    oss << "----------------------------------------\n";
    oss << "           Benchmark Results            \n";
    oss << "----------------------------------------\n";
    oss << "Model: " << model_name << "\n";
    oss << "Mean Squared Error (MSE): " << r.mse << "\n";
    oss << "Root Mean Squared Error (RMSE): " << r.rmse << "\n";
    oss << "R-squared: " << r.r2 << "\n";
    oss << "----------------------------------------\n";
    return oss.str();
}

void print_result(const BenchmarkResult& r, const std::string& model_name) {
    std::cout << format_result_pretty(r, model_name);
}

} // namespace mlsuite
