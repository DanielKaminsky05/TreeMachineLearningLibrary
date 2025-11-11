#ifndef REGRESSIONBENCHMARK_H
#define REGRESSIONBENCHMARK_H

#include "BenchmarkStrategy.h"

class RegressionBenchmark : public BenchmarkStrategy {
public:
    void execute(const IModel& model, const Dataset& testFeatures, const Dataset& testTargets) const override;
};

#endif // REGRESSIONBENCHMARK_H
