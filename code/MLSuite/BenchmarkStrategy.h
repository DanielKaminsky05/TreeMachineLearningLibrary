#ifndef BENCHMARKSTRATEGY_H
#define BENCHMARKSTRATEGY_H

#include <cstddef>
#include <string>
#include <limits>
#include <chrono>
#include "IModel.h"
#include "Dataset.h"

struct BenchmarkResult {
    std::string modelName;
    std::string taskType;          // "regression" or "classification"
    std::size_t numSamples{0};
    double fitMillis{0.0};
    double predictMillis{0.0};
    std::size_t memoryBytes{0};    // working set / current RSS snapshot

    // Regression metrics
    double mse{std::numeric_limits<double>::quiet_NaN()};
    double rmse{std::numeric_limits<double>::quiet_NaN()};
    double r2{std::numeric_limits<double>::quiet_NaN()};

    // Classification metrics
    double accuracy{std::numeric_limits<double>::quiet_NaN()};
    double precision{std::numeric_limits<double>::quiet_NaN()};
    double recall{std::numeric_limits<double>::quiet_NaN()};
    double f1{std::numeric_limits<double>::quiet_NaN()};
};

// Interface for benchmark strategies. Implementations should return a filled BenchmarkResult.
class BenchmarkStrategy {
public:
    virtual ~BenchmarkStrategy() = default;
    virtual BenchmarkResult execute(const IModel& model,
                                    const Dataset& xData,
                                    const Dataset& yData,
                                    double fitMillis = 0.0) const = 0;

    // Evaluates the model and returns a "loss" score (lower is better).
    // For Regression: Returns MSE.
    // For Classification: Returns (1.0 - Accuracy), i.e., Error Rate.
    virtual double evaluate(const IModel& model, 
                            const Dataset& features, 
                            const Dataset& targets) const = 0;

    // Orchestrates training (fit), timing, and executing the benchmark.
    BenchmarkResult trainAndExecute(IModel& model,
                                    const Dataset& trainFeatures,
                                    const Dataset& trainTargets,
                                    const Dataset& testFeatures,
                                    const Dataset& testTargets) const;
};

// Shared helpers for timing and memory snapshots.
double currentMemoryUsageBytes();
double millisBetween(const std::chrono::high_resolution_clock::time_point& start,
                     const std::chrono::high_resolution_clock::time_point& end);

#endif
