#ifndef LINEARREGRESSIONBUILDER_H
#define LINEARREGRESSIONBUILDER_H

#include "LinRegModel.h"
#include "Dataset.h"
#include <string>
#include <stdexcept>
#include <memory> // unique_ptr for returning ptr to the model 

class LinearRegressionBuilder {
public:
    	LinearRegressionBuilder();

    	LinearRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);
    	LinearRegressionBuilder& with_regularization(const std::string& type);
    	LinearRegressionBuilder& with_lambda(double lambda);

    	LinRegModel fit();

    	std::unique_ptr<LinRegModel> build_unfitted();

private:
    	Dataset* m_X_train;
    	Dataset* m_y_train;
    	std::string m_regularization;
    	double m_lambda;
};

#endif // LINEARREGRESSIONBUILDER_H
