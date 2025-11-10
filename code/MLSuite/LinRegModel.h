#ifndef LINREGMODEL_H
#define LINREGMODEL_H

#include "Dataset.h"
#include <Eigen/Dense>

class LinRegModel {
public:
    LinRegModel();
    void fit(Dataset& X_dataset, Dataset& y_dataset);
    Eigen::VectorXf predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test);
    Eigen::VectorXf get_theta();

private:
    Eigen::VectorXf m_theta;
};

#endif
