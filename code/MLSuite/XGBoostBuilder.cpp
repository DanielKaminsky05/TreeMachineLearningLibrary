#include "XGBoostBuilder.h"

XGBoostBuilder::XGBoostBuilder(int nEstCounts, float learningRate, const std::string& lossFn,
                               int depthValue, float ratioValue, float gammaValue,
                               const std::string& regularizationType)
    : n_est_counts(nEstCounts),
      learning_rate(learningRate),
      loss_fn(lossFn),
      depth(depthValue),
      ratio(ratioValue),
      gamma(gammaValue),
      regularization(regularizationType),
      model(nEstCounts, learningRate, lossFn, depthValue, ratioValue, gammaValue, regularizationType) {}

void XGBoostBuilder::set_model() {
    model = XGBoostModel(n_est_counts, learning_rate, loss_fn, depth, ratio, gamma, regularization);
}
