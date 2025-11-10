#include "LinRegModel.h"
#include "Dataset.h"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

LinRegModel::LinRegModel() {}

void LinRegModel::fit(Dataset& X_dataset, Dataset& y_dataset) {
    std::vector<float> x_data = X_dataset.get_data();
    std::vector<std::string> x_columns = X_dataset.get_columns();
    int n_cols_x = x_columns.size();
    int n_rows = x_data.size() / n_cols_x;

    std::vector<float> y_data = y_dataset.get_data();
    if (y_data.size() != n_rows) {
        throw std::invalid_argument("Number of rows in X and y datasets do not match.");
    }

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(x_data.data(), n_rows, n_cols_x);
    Eigen::Map<Eigen::VectorXf> y(y_data.data(), n_rows);

    Eigen::MatrixXf X_b(n_rows, n_cols_x + 1);
    X_b.setOnes();
    X_b.rightCols(n_cols_x) = X;

    m_theta = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
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
