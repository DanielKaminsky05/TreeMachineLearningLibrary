#ifndef REGRESSIONBENCHMARK_H
#define REGRESSIONBENCHMARK_H

#include "IBenchmarkStrategy.h"

class RegressionBenchmark : public IBenchmarkStrategy {
public:
    void execute(const IModel& model, const Dataset& testData) const override;
};

#endif // REGRESSIONBENCHMARK_H
