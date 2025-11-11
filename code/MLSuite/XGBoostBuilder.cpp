#include "XGBoostBuilder.h"

XGBoostBuilder::XGBoostBuilder(int nEstimatorsValue,
                               float learningRateValue,
                               int maxDepthValue,
                               float subsampleRatioValue,
                               float gammaValue,
                               const std::string& regularizationType)
    : nEstimators(nEstimatorsValue),
      learningRate(learningRateValue),
      maxDepth(maxDepthValue),
      subsampleRatio(subsampleRatioValue),
      gamma(gammaValue),
      regularization(regularizationType),
      model(nEstimatorsValue,
            learningRateValue,
            maxDepthValue,
            subsampleRatioValue,
            gammaValue,
            regularizationType) {}

void XGBoostBuilder::setModel() {
    model = XGBoostModel(nEstimators,
                         learningRate,
                         maxDepth,
                         subsampleRatio,
                         gamma,
                         regularization);
}
