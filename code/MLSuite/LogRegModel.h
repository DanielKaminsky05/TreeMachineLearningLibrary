// code/MLSuite/LogisticRegressionModel.h
#ifndef LOGREG_MODEL_H
#define LOGREG_MODEL_H

#include "IModel.h"
#include "Dataset.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>

class LogRegModel : public IModel {
public:
	LogRegModel();

    	void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values_vec, 
	      const std::string& regularization = "None", double lambda = 0.0, double learning_rate = 0.01, int num_iterations = 1000);

    	Eigen::VectorXf predict_proba(const Eigen::Ref<const Eigen::MatrixXf>& X_test) const;

    	Eigen::VectorXf predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test) const;

    	Eigen::VectorXf get_theta() const;

    	void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) override;
    	std::vector<float> predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const override;
    	std::string getName() const override;

private:
    	Eigen::VectorXf m_theta;
    	float sigmoid(float z) const;
};

#endif // LOGREG_MODEL_H

