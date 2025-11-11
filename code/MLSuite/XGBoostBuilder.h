#ifndef XGBOOSTBUILDER
#define XGBOOSTBUILDER

#include <string>
#include "XGBoostModel.h"

class XGBoostBuilder {
private:
    int n_est_counts;
    float learning_rate;
    std::string loss_fn;
    int depth;
    float ratio;
    float gamma;
    std::string regularization;
    XGBoostModel model;

public:
    XGBoostBuilder(int nEstCounts, float learningRate, const std::string& lossFn,
                   int depth, float ratio, float gamma, const std::string& regularization);

    void set_n_estimators(int count) {
        n_est_counts = count;
        model.set_n_estimators(count);
    }

    void set_learning_rate(float rate) {
        learning_rate = rate;
        model.set_learning_rate(rate);
    }

    void set_objective(const std::string& lossFn) {
        loss_fn = lossFn;
        model.set_objective(lossFn);
    }

    void set_max_depth(int depthValue) {
        depth = depthValue;
        model.set_max_depth(depthValue);
    }

    void set_subsample(float ratioValue) {
        ratio = ratioValue;
        model.set_subsample(ratioValue);
    }

    int get_n_estimators() const { return n_est_counts; }
    float get_learning_rate() const { return learning_rate; }
    const std::string& get_loss_fn() const { return loss_fn; }
    int get_depth() const { return depth; }
    float get_ratio() const { return ratio; }
    float get_gamma() const { return gamma; }
    const std::string& get_regularization() const { return regularization; }

    const XGBoostModel& get_model() const { return model; }
    XGBoostModel& get_model() { return model; }
    void set_model();
};

#endif
