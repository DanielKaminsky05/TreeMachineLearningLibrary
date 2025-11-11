#ifndef XGBOOSTBUILDER
#define XGBOOSTBUILDER

#include <string>
#include "XGBoostModel.h"

class XGBoostBuilder {
private:
    int nEstimators;
    float learningRate;
    int maxDepth;
    float subsampleRatio;
    float gamma;
    std::string regularization;
    XGBoostModel model;

public:
    XGBoostBuilder(int nEstimators, float learningRate, int maxDepth,
                   float subsampleRatio, float gamma, const std::string& regularization);

    void setNEstimators(int count) {
        nEstimators = count;
        model.setNEstimators(count);
    }

    void setLearningRate(float rate) {
        learningRate = rate;
        model.setLearningRate(rate);
    }

    void setMaxDepth(int depthValue) {
        maxDepth = depthValue;
        model.setMaxDepth(depthValue);
    }

    void setSubsampleRatio(float ratioValue) {
        subsampleRatio = ratioValue;
        model.setSubsampleRatio(ratioValue);
    }

    void setGamma(float gammaValue) {
        gamma = gammaValue;
        model.setGamma(gammaValue);
    }

    void setRegularization(const std::string& regularizationType) {
        regularization = regularizationType;
        model.setRegularization(regularizationType);
    }

    int getNEstimators() const { return nEstimators; }
    float getLearningRate() const { return learningRate; }
    int getDepth() const { return maxDepth; }
    float getSubsampleRatio() const { return subsampleRatio; }
    float getGamma() const { return gamma; }
    const std::string& getRegularization() const { return regularization; }

    const XGBoostModel& getModel() const { return model; }
    XGBoostModel& getModel() { return model; }
    void setModel();
};

#endif
