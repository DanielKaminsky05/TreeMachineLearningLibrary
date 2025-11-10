#ifndef LINEARREGRESSIONBUILDER_H
#define LINEARREGRESSIONBUILDER_H

#include "LinRegModel.h"
#include "Dataset.h"
#include <string>
#include <stdexcept>

class LinearRegressionBuilder {
public:
    // Constructor
    LinearRegressionBuilder();

    // "Setter" methods to configure the model, returning a reference to allow chaining
    LinearRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);
    LinearRegressionBuilder& with_regularization(const std::string& type);
    LinearRegressionBuilder& with_lambda(double lambda);

    // The final build/fit method that returns the trained model
    LinRegModel fit();

private:
    // Private members to store the configuration
    Dataset* m_X_train;
    Dataset* m_y_train;
    std::string m_regularization;
    double m_lambda;
};

#endif // LINEARREGRESSIONBUILDER_H
