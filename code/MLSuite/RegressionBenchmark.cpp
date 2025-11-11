#include "RegressionBenchmark.h"
#include "IModel.h"
#include "Dataset.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

// Helper function to calculate Mean Squared Error (MSE)
static double calculateMSE(const std::vector<float>& actual, const std::vector<float>& predicted) {
    double mse = 0.0;
    if (actual.empty()) return 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        mse += std::pow(actual[i] - predicted[i], 2);
    }
    return mse / actual.size();
}

// Helper function to calculate R-squared
static double calculateR2(const std::vector<float>& actual, const std::vector<float>& predicted) {
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
        // If total sum of squares is zero, R-squared is undefined or can be considered 1 if predictions are perfect.
        // For simplicity, returning 1 if residuals are also zero, otherwise 0.
        return (ss_res == 0.0) ? 1.0 : 0.0;
    }

    return 1.0 - (ss_res / ss_total);
}


void RegressionBenchmark::execute(const IModel& model, const Dataset& testData) const {
    std::cout << "\nExecuting regression benchmark..." << std::endl;

    // Note: I am assuming your model and dataset classes have these methods.
    // You may need to adjust these calls to match your actual class designs.
    std::vector<float> predictions = model.predict(testData.getFeatures());
    const std::vector<float>& actual = testData.getTargets();

    if (predictions.size() != actual.size()) {
        std::cerr << "Benchmark Error: Prediction size does not match actual size." << std::endl;
        return;
    }

    // Calculate metrics using helper functions
    double mse = calculateMSE(actual, predictions);
    double rmse = std::sqrt(mse);
    double r2 = calculateR2(actual, predictions);

    // Print results
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "           Benchmark Results            " << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Model: " << model.getName() << std::endl; // Assuming a getName() method on IModel
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << rmse << std::endl;
    std::cout << "R-squared: " << r2 << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}
