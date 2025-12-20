#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "IModel.h"
#include <vector>
#include <random>
#include "DecisionTree.h"

class RandomForest : public IModel {
    public:
	RandomForest(int Estimators, int maxDepth, int minSamplesSplit, int maxFeatures, bool bootstrap, int randomState, bool isClassification = false);
        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
        double predict(const std::vector<double>& X) const;
        std::vector<DecisionTree> getTrees() {return trees;};

	// IModel interface methods
	void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) override;
	std::vector<float> predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const override;
	std::string getName() const override;
    private:
        int nEstimators;
        int maxDepth;
	int minSamplesSplit;
        int maxFeatures;
        bool bootstrap;
        int randomState;
        bool isClassification;
        bool isFitted = false;
        int nFeatures = 0;
        std::vector<DecisionTree> trees;
        std::mt19937 internalRng;
        void buildTree(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
        std::vector<int> sampleBootstrap(int n);
        std::vector<int> sampleFeatures(int p, int maxFeatures);
        std::vector<std::vector<double>> predictAllTrees(const std::vector<std::vector<double>>& X);
        std::vector<double> aggregateMean(const std::vector<double>& preds);
};

#endif
