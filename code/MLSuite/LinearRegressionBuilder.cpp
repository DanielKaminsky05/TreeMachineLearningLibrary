#include "LinearRegressionBuilder.h"

// Constructor initializes with default values
LinearRegressionBuilder::LinearRegressionBuilder(): 
	m_X_train(nullptr), 
      	m_y_train(nullptr), 
      	m_regularization("None"), 
      	m_lambda(0.0) {}

// Sets the training data
LinearRegressionBuilder& LinearRegressionBuilder::with_training_data(Dataset& X_train, Dataset& y_train) {
	m_X_train = &X_train;
	m_y_train = &y_train;
	return *this;
}

// Sets the regularization type
LinearRegressionBuilder& LinearRegressionBuilder::with_regularization(const std::string& type) {
    	m_regularization = type;
    	return *this;
}

// Sets the lambda value
LinearRegressionBuilder& LinearRegressionBuilder::with_lambda(double lambda) {
	m_lambda = lambda;
    	return *this;
}

// Creates, trains, and returns the final model
LinRegModel LinearRegressionBuilder::fit() {
    // Ensure that the training data has been provided before fitting
	if (!m_X_train || !m_y_train) {
        	throw std::runtime_error("Training data must be provided before fitting the model.");
    	}

    	LinRegModel model;
    	model.fit(*m_X_train, *m_y_train, m_regularization, m_lambda);
    	return model;
}

// New method to build an unfitted model
std::unique_ptr<LinRegModel> LinearRegressionBuilder::build_unfitted() {
    return std::make_unique<LinRegModel>();
}
