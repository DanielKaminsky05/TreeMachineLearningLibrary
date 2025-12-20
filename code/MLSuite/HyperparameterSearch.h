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

    // random search is a pure virtual function, so all the derived classes must implement, in this case only classic model factory
    virtual std::unique_ptr<IModel> randomSearch(
        const std::string& modelType,
        const std::vector<std::vector<std::string>>& hyperParams,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const BenchmarkStrategy& evaluationStrategy) = 0;
};

#endif
