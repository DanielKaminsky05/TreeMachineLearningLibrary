#ifndef XGBOOSTMODEL_H
#define XGBOOSTMODEL_H

#include <string>
#include <vector>
#include "DecisionTree.h"
#include "IModel.h"

class XGBoostModel : public IModel {
private:
	int nEstimators;
	float learningRate;
	int maxDepth;
	float subsampleRatio;
    	float gamma;
    	std::string regularization;

    	std::vector<DecisionTree> trees;
    	double initialBias = 0.0;
    	bool isFitted = false;
        bool isClassification = false;

public:
	XGBoostModel(int nEstimators, float learningRate, int maxDepth, float subsampleRatio, float gamma, std::string regularization, bool isClassification = false);

    	double predict(const std::vector<double>& input) const;
    	void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);

    	void setNEstimators(int count) { nEstimators = count; }
    	void setLearningRate(float rate) { learningRate = rate; }
    	void setMaxDepth(int depthValue) { maxDepth = depthValue; }
    	void setSubsampleRatio(float ratio) { subsampleRatio = ratio; }
    	void setGamma(float gammaValue) { gamma = gammaValue; }
    	void setRegularization(const std::string& regularizationType) { regularization = regularizationType; }

    	int getNEstimators() const { return nEstimators; }
    	float getLearningRate() const { return learningRate; }
    	int getDepth() const { return maxDepth; }
    	float getSubsampleRatio() const { return subsampleRatio; }
    	float getGamma() const { return gamma; }
    	std::string getRegularization() const { return regularization; }

    	bool fitted() const { return isFitted; }
    	double bias() const { return initialBias; }

    	void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) override;
    	std::vector<float> predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const override;
    	std::string getName() const override { return "XGBoost"; }
};

#endif
