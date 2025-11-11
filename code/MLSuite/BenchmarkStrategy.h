#ifndef BENCHMARKSTRATEGY_H
#define BENCHMARKSTRATEGY_H

#include "IModel.h"
#include "Dataset.h"

class BenchmarkStrategy {
public:
    virtual ~BenchmarkStrategy() = default;
    virtual void execute(const IModel& model, const Dataset& testFeatures, const Dataset& testTargets) const = 0;
};

#endif 
