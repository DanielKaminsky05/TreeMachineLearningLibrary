#include "LinRegModel.h"
#include "Dataset.h"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

LinRegModel::LinRegModel() {}

void LinRegModel::fit(Dataset& X_dataset, Dataset& y_dataset, const std::string& regularization, double lambda) { 
    	std::vector<float> x_data = X_dataset.get_data();
    	std::vector<std::string> x_columns = X_dataset.get_columns();
    	int n_cols_x = x_columns.size();
    	int n_rows = x_data.size() / n_cols_x;

	std::vector<float> y_data = y_dataset.get_data();

    	if (y_data.size() != n_rows) {
        	throw std::invalid_argument("Number of rows in X and y datasets do not match.");
    	}

	if (regularization != "None" && regularization != "L2" && regularization != "L1") {
		throw std::invalid_argument("Invalid regularization type");

	}
	
    	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(x_data.data(), n_rows, n_cols_x);
    	Eigen::Map<Eigen::VectorXf> y(y_data.data(), n_rows);

    	Eigen::MatrixXf X_b(n_rows, n_cols_x + 1);
    	X_b.setOnes();
    	X_b.rightCols(n_cols_x) = X;

    	if (regularization == "L2") {
        	Eigen::MatrixXf I = Eigen::MatrixXf::Identity(n_cols_x + 1, n_cols_x + 1);
        	I(0, 0) = 0; // exclude bias 
        	m_theta = (X_b.transpose() * X_b + lambda * I).ldlt().solve(X_b.transpose() * y);
    	} else if (regularization == "L1") {
        	throw std::logic_error("L1 regularization requires an iterative solver and is not supported by this method.");
    	} else { // "None" or any other value
        	m_theta = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
    	}
}

Eigen::VectorXf LinRegModel::predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test) {
	Eigen::MatrixXf X_test_b(X_test.rows(), X_test.cols() + 1);
    	X_test_b.setOnes();
    	X_test_b.rightCols(X_test.cols()) = X_test;

    	return X_test_b * m_theta;
}

Eigen::VectorXf LinRegModel::get_theta() {
    	return m_theta;
}

// concrete method implementations for the IModel interface for benchmark 
void LinRegModel::fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) {

	if (x_values.empty() || y_values.empty() || columns.empty()) {
        	throw std::invalid_argument("Input vectors cannot be empty.");
    	}

    	size_t n_cols = columns.size();
    	size_t n_rows = x_values.size() / n_cols;

    	if (x_values.size() % n_cols != 0) {
        	throw std::invalid_argument("The size of x_values is not a multiple of the number of columns.");
    	}
    	if (n_rows != y_values.size()) {
    		throw std::invalid_argument("Number of samples in features and targets do not match.");
    	}

    	// map the 1D float vector to an Eigen Matrix
    	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(x_values.data(), n_rows, n_cols);

    	// map the target vector to an Eigen Vector
    	Eigen::Map<const Eigen::VectorXf> y(y_values.data(), n_rows);

    	// add bias term, solve theta for weights 
    	Eigen::MatrixXf X_b(n_rows, n_cols + 1);
    	X_b.setOnes();
    	X_b.rightCols(n_cols) = X;
    	m_theta = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
}

std::vector<float> LinRegModel::predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const {
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

    	if (m_theta.size() != n_cols + 1) {
         	throw std::invalid_argument("Number of features in prediction data does not match the trained model.");
    	}

    	// map the 1D float vector to an Eigen Matrix
    	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_test(x_values.data(), n_rows, n_cols);

    	// add bias term and predict
    	Eigen::MatrixXf X_test_b(X_test.rows(), X_test.cols() + 1);
    	X_test_b.setOnes();
    	X_test_b.rightCols(X_test.cols()) = X_test;
    	Eigen::VectorXf predictions_eigen = X_test_b * m_theta;

    	// convert Eigen::VectorXf back to std::vector<float> after done with Eigen 
    	return std::vector<float>(predictions_eigen.data(), predictions_eigen.data() + predictions_eigen.size());
}

std::string LinRegModel::getName() const {
	return "Linear Regression";
}
