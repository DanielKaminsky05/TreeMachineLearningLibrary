#ifndef REGRESSIONBENCHMARK_H
#define REGRESSIONBENCHMARK_H

#include "BenchmarkStrategy.h"

class RegressionBenchmark : public BenchmarkStrategy {
public:
    BenchmarkResult execute(const IModel& model,
                            const Dataset& xData,
                            const Dataset& actualData,
                            double fitMillis = 0.0) const override;

    double evaluate(const IModel& model, 
                    const Dataset& features, 
                    const Dataset& targets) const override;
};

#endif // REGRESSIONBENCHMARK_H
