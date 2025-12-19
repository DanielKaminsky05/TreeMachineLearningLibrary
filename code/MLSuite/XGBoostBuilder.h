#ifndef XGBOOSTBUILDER_H
#define XGBOOSTBUILDER_H

#include <string>
#include <memory>
#include "XGBoostModel.h"

class XGBoostBuilder {
public:
	XGBoostBuilder();

	XGBoostBuilder& setNEstimators(int count);
    	XGBoostBuilder& setLearningRate(float rate);
    	XGBoostBuilder& setMaxDepth(int depthValue);
    	XGBoostBuilder& setSubsampleRatio(float ratioValue);
    	XGBoostBuilder& setGamma(float gammaValue);
    	XGBoostBuilder& setRegularization(const std::string& regularizationType);
        XGBoostBuilder& setIsClassification(bool isClassification);

    	std::unique_ptr<XGBoostModel> build();

private:
    	int nEstimators;
    	float learningRate;
    	int maxDepth;
    	float subsampleRatio;
    	float gamma;
    	std::string regularization;
        bool isClassification = false;
};

#endif 
