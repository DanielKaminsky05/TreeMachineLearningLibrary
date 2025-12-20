#ifndef LOGISTIC_REGRESSION_BUILDER_H
#define LOGISTIC_REGRESSION_BUILDER_H

#include "LogRegModel.h"
#include "Dataset.h"
#include <string>
#include <memory>

class LogisticRegressionBuilder {
public:
    LogisticRegressionBuilder();

    LogisticRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);

    LogisticRegressionBuilder& with_regularization(const std::string& type);

    LogisticRegressionBuilder& with_lambda(double lambda);

    LogisticRegressionBuilder& with_learning_rate(double rate);

    LogisticRegressionBuilder& with_num_iterations(int iterations);

    LogRegModel fit();

    std::unique_ptr<LogRegModel> build_unfitted();

private:
    Dataset* m_X_train;
    Dataset* m_y_train;
    std::string m_regularization;
    double m_lambda;
    double m_learning_rate;
    int m_num_iterations;
};

#endif // LOGISTIC_REGRESSION_BUILDER_H
