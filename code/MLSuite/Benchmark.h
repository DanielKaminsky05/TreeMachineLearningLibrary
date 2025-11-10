#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include <string>

class Benchmark {
public:
    virtual ~Benchmark() = default; // Virtual destructor for proper cleanup of derived classes

    // Pure virtual function for calculating metrics like MAE, MSE, RMSE
    virtual double calculate_metrics(const std::vector<float>& y_test, 
                                     const std::vector<float>& predictions, 
                                     const std::string& metric) = 0;

    // Pure virtual function for evaluating models with metrics like R squared score, accuracy
    virtual double evaluate(const std::vector<float>& y_test, 
                            const std::vector<float>& predictions, 
                            const std::string& method) = 0;
};

#endif // BENCHMARK_H
