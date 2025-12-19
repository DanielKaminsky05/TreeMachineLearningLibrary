#ifndef HYPERPARAMETERSEARCH_H
#define HYPERPARAMETERSEARCH_H

#include "IModel.h"
#include "BenchmarkStrategy.h" // Required for strategy pattern
#include <memory>
#include <vector>
#include <string>

class HyperparameterSearch {
public:
    virtual ~HyperparameterSearch() = default;

    // Pure virtual: derived classes must implement
    virtual std::unique_ptr<IModel> randomSearch(
        const std::string& modelType,
        const std::vector<std::vector<std::string>>& hyperParams,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const BenchmarkStrategy& evaluationStrategy) = 0;
};

#endif