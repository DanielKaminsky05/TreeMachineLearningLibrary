#include "XGBoostBuilder.h"

XGBoostBuilder::XGBoostBuilder()
    : nEstimators(100),
      learningRate(0.1f),
      maxDepth(3),
      subsampleRatio(1.0f),
      gamma(0.0f),
      regularization("L2"),
      isClassification(false) {}

XGBoostBuilder& XGBoostBuilder::setNEstimators(int count) {
	nEstimators = count;
	return *this;
}

XGBoostBuilder& XGBoostBuilder::setLearningRate(float rate) {
	learningRate = rate;
    	return *this;
}

XGBoostBuilder& XGBoostBuilder::setMaxDepth(int depthValue) {
    	maxDepth = depthValue;
    	return *this;
}

XGBoostBuilder& XGBoostBuilder::setSubsampleRatio(float ratioValue) {
	subsampleRatio = ratioValue;
    	return *this;
}

XGBoostBuilder& XGBoostBuilder::setGamma(float gammaValue) {
    	gamma = gammaValue;
    	return *this;
}

XGBoostBuilder& XGBoostBuilder::setRegularization(const std::string& regularizationType) {
    	regularization = regularizationType;
    	return *this;
}

XGBoostBuilder& XGBoostBuilder::setIsClassification(bool isClassificationValue) {
    isClassification = isClassificationValue;
    return *this;
}

std::unique_ptr<XGBoostModel> XGBoostBuilder::build() {
    	return std::make_unique<XGBoostModel>(nEstimators, learningRate, maxDepth, subsampleRatio, gamma, regularization, isClassification);
}