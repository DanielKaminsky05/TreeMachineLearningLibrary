#include "RegressionBenchmark.h"
#include "IModel.h"
#include "Dataset.h"
#include "BenchmarkStrategy.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

// contains all methods to benchmark performance metrics of a regression model, concrete Strategy implementation for BenchmarkStrategy
namespace {

double calculateMSE(const std::vector<float>& actual, const std::vector<float>& predicted) {
    double mse = 0.0;
    if (actual.empty()) return 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        mse += std::pow(actual[i] - predicted[i], 2);
    }
    return mse / actual.size();
}

double calculateR2(const std::vector<float>& actual, const std::vector<float>& predicted) {
    if (actual.empty()) return 0.0;
    double sum_actual = std::accumulate(actual.begin(), actual.end(), 0.0);
    double mean_actual = sum_actual / actual.size();

    double ss_total = 0.0;
    double ss_res = 0.0;

    for (size_t i = 0; i < actual.size(); ++i) {
        ss_total += std::pow(actual[i] - mean_actual, 2);
        ss_res += std::pow(actual[i] - predicted[i], 2);
    }

    if (ss_total == 0.0) {
	return (ss_res == 0.0) ? 1.0 : 0.0;
    }

    return 1.0 - (ss_res / ss_total);
}
} // namespace

BenchmarkResult RegressionBenchmark::execute(const IModel& model,
                                             const Dataset& xData,
                                             const Dataset& actualData,
                                             double fitMillis) const {
    BenchmarkResult result;
    result.modelName = model.getName();
    result.taskType = "regression";
    result.numSamples = actualData.get_data().size();
    result.fitMillis = fitMillis;

    const auto startPredict = std::chrono::high_resolution_clock::now();
    std::vector<float> predictions = model.predict(xData.get_data(), xData.get_columns());
    const auto endPredict = std::chrono::high_resolution_clock::now();
    result.predictMillis = millisBetween(startPredict, endPredict);

    result.memoryBytes = static_cast<std::size_t>(currentMemoryUsageBytes());

    const std::vector<float>& actual = actualData.get_data();

    if (predictions.size() != actual.size()) {
        std::cerr << "Benchmark Error: Prediction size does not match actual size." << std::endl;
        return result;
    }

    result.mse = calculateMSE(actual, predictions);
    result.rmse = std::sqrt(result.mse);
    result.r2 = calculateR2(actual, predictions);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "       Regression Benchmark Results     " << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Model: " << result.modelName << std::endl;
    std::cout << "Samples: " << result.numSamples << std::endl;
    std::cout << "Fit time (ms): " << result.fitMillis << std::endl;
    std::cout << "Predict time (ms): " << result.predictMillis << std::endl;
    std::cout << "Memory (bytes): " << result.memoryBytes << std::endl;
    std::cout << "MSE: " << result.mse << std::endl;
    std::cout << "RMSE: " << result.rmse << std::endl;
    std::cout << "R-squared: " << result.r2 << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    return result;
}

double RegressionBenchmark::evaluate(const IModel& model, 
                                     const Dataset& features, 
                                     const Dataset& targets) const {
    std::vector<float> predictions = model.predict(features.get_data(), features.get_columns());
    const std::vector<float>& actual = targets.get_data();

    if (predictions.size() != actual.size()) { // size safety check 
        std::cerr << "evaluate: Prediction size mismatch." << std::endl;
        return std::numeric_limits<double>::infinity(); 
    }

    return calculateMSE(actual, predictions);
}
