#ifndef LINREGMODEL_H
#define LINREGMODEL_H

#include "Dataset.h"
#include "IModel.h" 
#include <Eigen/Dense>
#include <string>
#include <vector>

class LinRegModel : public IModel { // implement IModel for it to work with classic model factory 
public:
	LinRegModel();

    	void fit(Dataset& X_dataset, Dataset& y_dataset, const std::string& regularization = "None", double lambda = 0.1);
    	Eigen::VectorXf predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test);
    	Eigen::VectorXf get_theta();

    	// IModel method 
    	void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) override;
    	std::vector<float> predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const override;
    	std::string getName() const override;

private:
    	Eigen::VectorXf m_theta;
};

#endif
