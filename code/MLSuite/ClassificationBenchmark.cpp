#include "ClassificationBenchmark.h"
#include "Dataset.h"
#include "IModel.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>

// benchmark results for classif models
namespace {
double safeDiv(double num, double den) {
	if (den == 0.0) return std::numeric_limits<double>::quiet_NaN();
    	return num / den;
}

struct ConfusionCounts {
    	double tp{0}, fp{0}, fn{0}, tn{0};
};

ConfusionCounts computeBinaryConfusion(const std::vector<int>& actual, const std::vector<int>& predicted, int positiveLabel = 1) {
    	ConfusionCounts c;

    	for (std::size_t i = 0; i < actual.size(); ++i) {
        	bool isPos = actual[i] == positiveLabel;
        	bool predPos = predicted[i] == positiveLabel;
        	if (isPos && predPos) ++c.tp;
        	else if (!isPos && predPos) ++c.fp;
        	else if (isPos && !predPos) ++c.fn;
        	else ++c.tn;
    	}

    	return c;
}

double computeAccuracy(const std::vector<int>& actual, const std::vector<int>& predicted) {
    	if (actual.empty()) return std::numeric_limits<double>::quiet_NaN();
    	double correct = 0.0;
    	for (std::size_t i = 0; i < actual.size(); ++i) {
        	if (actual[i] == predicted[i]) {
            		++correct;
        	}
    	}

    return correct / static_cast<double>(actual.size());
}

} // namespace

BenchmarkResult ClassificationBenchmark::execute(const IModel& model, const Dataset& xData, const Dataset& yData, double fitMillis) const {
	BenchmarkResult result;
    	result.modelName = model.getName();
    	result.taskType = "classification";
    	result.numSamples = yData.get_data().size();
    	result.fitMillis = fitMillis;

    	const auto startPredict = std::chrono::high_resolution_clock::now();
    	std::vector<float> rawPreds = model.predict(xData.get_data(), xData.get_columns());
    	const auto endPredict = std::chrono::high_resolution_clock::now();
    	result.predictMillis = millisBetween(startPredict, endPredict);
    	result.memoryBytes = static_cast<std::size_t>(currentMemoryUsageBytes());

    	const std::vector<float>& actualRaw = yData.get_data();

    	if (rawPreds.size() != actualRaw.size()) {
        	std::cerr << "Benchmark Error: Prediction size does not match actual size." << std::endl;
        	return result;
    	}

    	std::vector<int> actual(actualRaw.size());
    	std::vector<int> predicted(rawPreds.size());

    	for (std::size_t i = 0; i < actualRaw.size(); ++i) {
        	actual[i] = static_cast<int>(std::round(actualRaw[i]));
        	predicted[i] = static_cast<int>(std::round(rawPreds[i]));
    	}

    	result.accuracy = computeAccuracy(actual, predicted);

    	// Basic binary precision/recall/f1 assuming positive label == 1
    	ConfusionCounts c = computeBinaryConfusion(actual, predicted, 1);
    	result.precision = safeDiv(c.tp, c.tp + c.fp);
    	result.recall = safeDiv(c.tp, c.tp + c.fn);
    	const double denom = (result.precision + result.recall);
    	result.f1 = (std::isnan(denom) || denom == 0.0)
                    ? std::numeric_limits<double>::quiet_NaN()
                    : 2.0 * result.precision * result.recall / denom;

    	std::cout << "----------------------------------------" << std::endl;
    	std::cout << "    Classification Benchmark Results    " << std::endl;
    	std::cout << "----------------------------------------" << std::endl;
    	std::cout << "Model: " << result.modelName << std::endl;
    	std::cout << "Samples: " << result.numSamples << std::endl;
    	std::cout << "Fit time (ms): " << result.fitMillis << std::endl;
    	std::cout << "Predict time (ms): " << result.predictMillis << std::endl;
    	std::cout << "Memory (bytes): " << result.memoryBytes << std::endl;
    	std::cout << "Accuracy: " << result.accuracy << std::endl;
    	std::cout << "Precision: " << result.precision << std::endl;
    	std::cout << "Recall: " << result.recall << std::endl;
    	std::cout << "F1: " << result.f1 << std::endl;
    	std::cout << "----------------------------------------" << std::endl;

    	return result;
}

double ClassificationBenchmark::evaluate(const IModel& model, const Dataset& features, const Dataset& targets) const {

    	std::vector<float> rawPreds = model.predict(features.get_data(), features.get_columns());
    	const std::vector<float>& actualRaw = targets.get_data();

    	if (rawPreds.size() != actualRaw.size()) {
        	std::cerr << "evaluate: Prediction size mismatch." << std::endl;
        	return std::numeric_limits<double>::infinity(); 
    	}

    	std::vector<int> actual(actualRaw.size());
    	std::vector<int> predicted(rawPreds.size());

    	for (std::size_t i = 0; i < actualRaw.size(); ++i) {
        	actual[i] = static_cast<int>(std::round(actualRaw[i]));
        	predicted[i] = static_cast<int>(std::round(rawPreds[i]));
    	}

    	// Optimization goal: minimize error rate (1.0 - accuracy)
    	double accuracy = computeAccuracy(actual, predicted);
    	return 1.0 - accuracy;
}
