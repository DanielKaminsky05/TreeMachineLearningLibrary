#ifndef CLASSIFICATIONBENCHMARK_H
#define CLASSIFICATIONBENCHMARK_H

#include "BenchmarkStrategy.h"

class ClassificationBenchmark : public BenchmarkStrategy {
public:
    BenchmarkResult execute(const IModel& model,
                            const Dataset& xData,
                            const Dataset& yData,
                            double fitMillis = 0.0) const override;

    double evaluate(const IModel& model, 
                    const Dataset& features, 
                    const Dataset& targets) const override;
};

#endif // CLASSIFICATIONBENCHMARK_H
