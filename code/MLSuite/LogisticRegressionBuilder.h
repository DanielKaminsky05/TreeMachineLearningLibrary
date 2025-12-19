// code/MLSuite/LogisticRegressionBuilder.h
#ifndef LOGISTIC_REGRESSION_BUILDER_H
#define LOGISTIC_REGRESSION_BUILDER_H

#include "LogRegModel.h"
#include "Dataset.h"
#include <string>
#include <memory>

class LogisticRegressionBuilder {
public:
    LogisticRegressionBuilder();

    // Sets the training data
    LogisticRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);

    // Sets the regularization type ("None", "L1", "L2")
    LogisticRegressionBuilder& with_regularization(const std::string& type);

    // Sets the lambda value for regularization
    LogisticRegressionBuilder& with_lambda(double lambda);

    // Sets the learning rate for gradient descent
    LogisticRegressionBuilder& with_learning_rate(double rate);

    // Sets the number of iterations for gradient descent
    LogisticRegressionBuilder& with_num_iterations(int iterations);

    // Creates, trains, and returns the final model
    LogRegModel fit();

    // Builds an unfitted model
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
