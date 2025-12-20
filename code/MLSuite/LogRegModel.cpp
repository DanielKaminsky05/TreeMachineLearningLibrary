#include "LogRegModel.h"
#include <cmath> 
#include <iostream> 

LogRegModel::LogRegModel() {}

float LogRegModel::sigmoid(float z) const {
	return 1.0f / (1.0f + std::exp(-z));
}

// fit method 
void LogRegModel::fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values_vec,
		      const std::string& regularization, double lambda, double learning_rate, int num_iterations) {
    
    	if (x_values.empty() || y_values_vec.empty() || columns.empty()) {
        	throw std::invalid_argument("Input vectors cannot be empty.");
    	}

    	size_t n_cols = columns.size();
    	size_t n_rows = x_values.size() / n_cols;

    	if (x_values.size() % n_cols != 0) {
        	throw std::invalid_argument("The size of x_values is not a multiple of the number of columns.");
    	}

    	if (n_rows != y_values_vec.size()) {
        	throw std::invalid_argument("Number of samples in features and targets do not match.");
    	}

    	if (regularization != "None" && regularization != "L2" && regularization != "L1") {
        	throw std::invalid_argument("Invalid regularization type. Must be 'None', 'L1', or 'L2'.");
    	}

    	if (regularization == "L1") {
        	throw std::logic_error("L1 regularization for Logistic Regression requires an iterative solver with subgradient methods and is not supported by this implementation.");
    	}

    	if (learning_rate <= 0) {
        	throw std::invalid_argument("Learning rate must be positive.");
    	}

    	if (num_iterations <= 0) {
        	throw std::invalid_argument("Number of iterations must be positive.");
    	}
    
    	// map the 1D float vector to an Eigen Matrix
    	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(x_values.data(), n_rows, n_cols);
    	Eigen::Map<const Eigen::VectorXf> y(y_values_vec.data(), n_rows); // map the target vector to an Eigen Vector

    	Eigen::MatrixXf X_b(n_rows, n_cols + 1);
    	X_b.setOnes();
    	X_b.rightCols(n_cols) = X;

    	// make weights (theta) with zeros
    	m_theta = Eigen::VectorXf::Zero(n_cols + 1);

    	for (int i = 0; i < num_iterations; ++i) {
        	Eigen::VectorXf z = X_b * m_theta;
        	Eigen::VectorXf h = z.unaryExpr([this](float val){ return sigmoid(val); }); // Predicted probabilities

        	Eigen::VectorXf error = h - y;
        	Eigen::VectorXf gradient = X_b.transpose() * error / static_cast<float>(n_rows);

        	// regularization
        	if (regularization == "L2") {
        	// ignore the bias term (m_theta(0))
            	Eigen::VectorXf regularized_theta = m_theta;
            	regularized_theta(0) = 0.0; 
            	gradient += (lambda / static_cast<float>(n_rows)) * regularized_theta;
        }

        m_theta -= learning_rate * gradient;
    }
}

Eigen::VectorXf LogRegModel::predict_proba(const Eigen::Ref<const Eigen::MatrixXf>& X_test) const {

	if (m_theta.size() == 0) {
        	throw std::logic_error("Model has not been fitted yet. Call fit() before predict_proba().");
    	}

    	if (X_test.cols() + 1 != m_theta.size()) {
        	throw std::invalid_argument("Number of features in prediction data does not match the trained model.");
    	}

    	Eigen::MatrixXf X_test_b(X_test.rows(), X_test.cols() + 1);
    	X_test_b.setOnes();
    	X_test_b.rightCols(X_test.cols()) = X_test;

    	Eigen::VectorXf z = X_test_b * m_theta;
    	return z.unaryExpr([this](float val){ return sigmoid(val); });
}

Eigen::VectorXf LogRegModel::predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test) const {
	Eigen::VectorXf probabilities = predict_proba(X_test);
    	Eigen::VectorXf predictions = Eigen::VectorXf::Zero(probabilities.size());

    	for (int i = 0; i < probabilities.size(); ++i) {
        	predictions(i) = (probabilities(i) >= 0.5f) ? 1.0f : 0.0f;
    	}

	return predictions;
}

Eigen::VectorXf LogRegModel::get_theta() const {
	return m_theta;
}

// concrete method implementations for the IModel interface for benchmark, overload now calls the primary fit method with default hyperparameters
void LogRegModel::fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) {
	fit(x_values, columns, y_values, "None", 0.0, 0.01, 1000); 
}

std::vector<float> LogRegModel::predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const {
	if (x_values.empty() || columns.empty()) {
        	return {};
    	}

    	if (m_theta.size() == 0) {
        	throw std::logic_error("Model has not been fitted yet. Call fit() before predict().");
    	}

    	size_t n_cols = columns.size();
    	size_t n_rows = x_values.size() / n_cols;

    	if (x_values.size() % n_cols != 0) {
        	throw std::invalid_argument("The size of x_values is not a multiple of the number of columns.");
    	}
    
    	if (m_theta.size() != n_cols + 1) { // +1 for bias term 
        	throw std::invalid_argument("Number of features in prediction data does not match the trained model.");
    	}

    	// map 1D float vector to an Eigen Matrix
    	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_test_eigen(x_values.data(), n_rows, n_cols);

    	// get preds 
    	Eigen::VectorXf predictions_eigen = predict(X_test_eigen);

    	// Eigen::VectorXf back to std::vector<float>
    	return std::vector<float>(predictions_eigen.data(), predictions_eigen.data() + predictions_eigen.size());
}

// get name for benchmarkstrategy 
std::string LogRegModel::getName() const {
    	return "Logistic Regression";
}
