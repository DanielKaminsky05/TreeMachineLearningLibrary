#include "LogisticRegressionBuilder.h"
#include <stdexcept>

LogisticRegressionBuilder::LogisticRegressionBuilder(): 
	m_X_train(nullptr), 
	m_y_train(nullptr), 
	m_regularization("None"), 
	m_lambda(0.0),
	m_learning_rate(0.01),    
    	m_num_iterations(1000) {} 

// setters for data, regularization, learning rate, lambda, number of iterations 
LogisticRegressionBuilder& LogisticRegressionBuilder::with_training_data(Dataset& X_train, Dataset& y_train) {
	m_X_train = &X_train;
    	m_y_train = &y_train;

    	return *this;
}


LogisticRegressionBuilder& LogisticRegressionBuilder::with_regularization(const std::string& type) {
	m_regularization = type;
    	return *this;
}

LogisticRegressionBuilder& LogisticRegressionBuilder::with_lambda(double lambda) {
	m_lambda = lambda;
	return *this;
}

LogisticRegressionBuilder& LogisticRegressionBuilder::with_learning_rate(double rate) {
	m_learning_rate = rate;
	return *this;
}

LogisticRegressionBuilder& LogisticRegressionBuilder::with_num_iterations(int iterations) {
	m_num_iterations = iterations;
	return *this;
}

// train and return final model 
LogRegModel LogisticRegressionBuilder::fit() {
	if (!m_X_train || !m_y_train) {
        	throw std::runtime_error("Training data must be provided before fitting the model.");
    	}

    	LogRegModel model;

    	std::vector<float> x_data = m_X_train->get_data();
    	std::vector<std::string> x_cols = m_X_train->get_columns();
    	std::vector<float> y_data = m_y_train->get_data();

    	model.fit(x_data, x_cols, y_data, m_regularization, m_lambda, m_learning_rate, m_num_iterations);
    	return model;
}

// return untrained model 
std::unique_ptr<LogRegModel> LogisticRegressionBuilder::build_unfitted() {
    return std::make_unique<LogRegModel>();
}
