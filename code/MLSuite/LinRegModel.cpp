#include "LinRegModel.h"
#include "Dataset.h"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

LinRegModel::LinRegModel() {}

void LinRegModel::fit(Dataset& X_dataset, Dataset& y_dataset, const std::string& regularization, double lambda) { // TODO: add order+ API for linreg.
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

// --- IModel Interface Implementation ---

// Adapter for the fit method.
// NOTE: This re-implements the fitting logic for the "None" regularization case.
// This is necessary because the original fit() method takes non-const Dataset references,
// which is incompatible with the data provided by the IModel interface.
void LinRegModel::fit(const std::vector<std::vector<float>>& features, const std::vector<float>& targets) {
    if (features.empty() || targets.empty()) {
        throw std::invalid_argument("Input features and targets cannot be empty.");
    }
    if (features.size() != targets.size()) {
        throw std::invalid_argument("Number of samples in features and targets do not match.");
    }

    size_t n_rows = features.size();
    size_t n_cols = features[0].size();

    // Convert std::vector<std::vector<float>> to Eigen::MatrixXf
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(n_rows, n_cols);
    for(size_t i = 0; i < n_rows; ++i) {
        if (features[i].size() != n_cols) {
            throw std::invalid_argument("All feature vectors must have the same number of columns.");
        }
        X.row(i) = Eigen::Map<const Eigen::RowVectorXf>(features[i].data(), n_cols);
    }

    // Convert std::vector<float> to Eigen::VectorXf
    Eigen::Map<const Eigen::VectorXf> y(targets.data(), n_rows);

    // Add bias term and solve for theta (standard normal equation)
    Eigen::MatrixXf X_b(n_rows, n_cols + 1);
    X_b.setOnes();
    X_b.rightCols(n_cols) = X;
    m_theta = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
}

// Adapter for the predict method.
// NOTE: This re-implements the prediction logic because the original predict() method
// was not marked as 'const', making it uncallable from this const interface method.
std::vector<float> LinRegModel::predict(const std::vector<std::vector<float>>& features) const {
    if (features.empty()) {
        return {};
    }
    if (m_theta.size() == 0) {
        throw std::logic_error("Model has not been fitted yet. Call fit() before predict().");
    }

    size_t n_rows = features.size();
    size_t n_cols = features[0].size();

    if (m_theta.size() != n_cols + 1) {
         throw std::invalid_argument("Number of features in prediction data does not match the trained model.");
    }

    // Convert input vector to Eigen Matrix
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_test(n_rows, n_cols);
    for(size_t i = 0; i < n_rows; ++i) {
        if (features[i].size() != n_cols) {
            throw std::invalid_argument("All feature vectors must have the same number of columns.");
        }
        X_test.row(i) = Eigen::Map<const Eigen::RowVectorXf>(features[i].data(), n_cols);
    }

    // prediction
    Eigen::MatrixXf X_test_b(X_test.rows(), X_test.cols() + 1);
    X_test_b.setOnes();
    X_test_b.rightCols(X_test.cols()) = X_test;
    Eigen::VectorXf predictions_eigen = X_test_b * m_theta;

    // Convert Eigen::VectorXf back to std::vector<float>
    return std::vector<float>(predictions_eigen.data(), predictions_eigen.data() + predictions_eigen.size());
}

std::string LinRegModel::getName() const {
    return "Linear Regression";
}
