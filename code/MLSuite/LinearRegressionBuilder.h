#ifndef LINEARREGRESSIONBUILDER_H
#define LINEARREGRESSIONBUILDER_H

#include "LinRegModel.h"
#include "Dataset.h"
#include <string>
#include <stdexcept>
#include <memory> // For std::unique_ptr

class LinearRegressionBuilder {
public:
    // Constructor
    LinearRegressionBuilder();

    // "Setter" methods to configure the model, returning a reference to allow chaining
    LinearRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);
    LinearRegressionBuilder& with_regularization(const std::string& type);
    LinearRegressionBuilder& with_lambda(double lambda);

    // The original build/fit method that returns the trained model
    LinRegModel fit();

    // New method to build an unfitted model
    std::unique_ptr<LinRegModel> build_unfitted();

private:
    // Private members to store the configuration
    Dataset* m_X_train;
    Dataset* m_y_train;
    std::string m_regularization;
    double m_lambda;
};

#endif // LINEARREGRESSIONBUILDER_H
