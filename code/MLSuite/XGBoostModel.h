#ifndef XGBOOSTMODEL_H
#define XGBOOSTMODEL_H

#include <string>
#include <vector>
#include "DecisionTree.h"

class XGBoostModel {
private:
    // Hyperparameters
    int n_est_counts;
    float learning_rate;
    std::string loss_fn;           // "reg:squarederror" or "binary:logistic"
    int depth;
    float subsample_ratio;
    float gamma;                   // (kept for API completeness; not used by this simple DT)
    std::string regularization;    // (kept for API completeness; not used by this simple DT)

    // Model state
    std::vector<DecisionTree> trees;
    double init_bias = 0.0;        // mean(y) for regression; logit(pos_rate) for logistic
    bool is_fitted = false;

    // Helpers
    static double sigmoid(double z);
    static double logit(double p);
    static double clipped(double x, double lo, double hi);

public:
    // Constructor (declared only; implemented in .cpp)
    XGBoostModel(int n_est_counts, float learning_rate, std::string loss_fn,
                 int depth, float subsample_ratio, float gamma, std::string regularization);

    // Core API
    double predict(const std::vector<double> input);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);

    // ---- Setters ----
    void set_n_estimators(int count) { n_est_counts = count; }
    void set_learning_rate(float rate) { learning_rate = rate; }
    void set_objective(const std::string& loss) { loss_fn = loss; }
    void set_max_depth(int d) { depth = d; }
    void set_subsample(float ratio) { subsample_ratio = ratio; }

    // ---- Getters ----
    int get_n_estimators() const { return n_est_counts; }
    float get_learning_rate() const { return learning_rate; }
    std::string get_loss_fn() const { return loss_fn; }
    int get_depth() const { return depth; }
    float get_ratio() const { return subsample_ratio; }
    float get_gamma() const { return gamma; }
    std::string get_regularization() const { return regularization; }

    // Optional: access to internals
    bool fitted() const { return is_fitted; }
    double bias() const { return init_bias; }
};

#endif
