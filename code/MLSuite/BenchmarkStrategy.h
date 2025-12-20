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
    	std::size_t memoryBytes{0};    // current RSS snapshots, working set 

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

// benchmark strategy, impls return a filled BenchmarkResult
class BenchmarkStrategy {
public:
	virtual ~BenchmarkStrategy() = default;
    	virtual BenchmarkResult execute(const IModel& model, const Dataset& xData, const Dataset& yData, double fitMillis = 0.0) const = 0;

    	// MSE for regression, 1 - accuracy for classif 
    	virtual double evaluate(const IModel& model, const Dataset& features, const Dataset& targets) const = 0;

    	// train, time and execute 
    	BenchmarkResult trainAndExecute(IModel& model, const Dataset& trainFeatures, const Dataset& trainTargets, const Dataset& testFeatures, 
				     const Dataset& testTargets) const;
};

// shared helpers for timing and memory snapshots
double currentMemoryUsageBytes();
double millisBetween(const std::chrono::high_resolution_clock::time_point& start, const std::chrono::high_resolution_clock::time_point& end);

#endif
