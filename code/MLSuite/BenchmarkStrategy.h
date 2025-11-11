// BenchmarkStrategy.h
#ifndef MLSUITE_BENCHMARK_STRATEGY_H
#define MLSUITE_BENCHMARK_STRATEGY_H

#include <vector>
#include <string>
#include <functional>

// Forward declaration to avoid hard dependency in header
class DecisionTree;

namespace mlsuite {

struct BenchmarkResult {
    // Dataset stats
    size_t n_train_samples = 0;
    size_t n_train_features = 0;
    size_t n_test_samples = 0;
    size_t n_test_features = 0;

    // Timing (seconds)
    double train_seconds = 0.0;
    double predict_seconds = 0.0;

    // Metrics
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;

    // Optional: placeholder if you later add memory tracking
    long peak_memory_kb = -1; // -1 means not measured

    // Utility helpers
    std::string toString() const;
};

// Strategy for regression benchmarks (MSE, RMSE, MAE, R^2, runtime)
class RegressionBenchmarkStrategy {
public:
    RegressionBenchmarkStrategy() = default;

    // Run with generic callbacks (keeps this independent of concrete model types).
    // The predict_fn must capture test inputs (X_test) and return predictions for it.
    BenchmarkResult run_with_functions(
        const std::function<void()>& train_fn,
        const std::function<std::vector<double>()>& predict_fn,
        const std::vector<double>& y_true,
        size_t n_train_samples,
        size_t n_train_features,
        size_t n_test_samples,
        size_t n_test_features
    ) const;

    // Convenience: run directly on a DecisionTree (available in this repo).
    BenchmarkResult run_decision_tree(
        DecisionTree& model,
        const std::vector<std::vector<double>>& X_train,
        const std::vector<double>& y_train,
        const std::vector<std::vector<double>>& X_test,
        const std::vector<double>& y_test
    ) const;

private:
    // Metric helpers
    static double calc_mse(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static double calc_mae(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static double calc_r2(const std::vector<double>& y_true, const std::vector<double>& y_pred);
};

// Pretty formatter and printer matching the requested output style
std::string format_result_pretty(const BenchmarkResult& r, const std::string& model_name);
void print_result(const BenchmarkResult& r, const std::string& model_name);

} // namespace mlsuite

#endif // MLSUITE_BENCHMARK_STRATEGY_H
