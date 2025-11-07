#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H


#include <vector>
#include <random>
#include "DecisionTree.h"

class RandomForest {
    private:
        int nEstimators;
        int maxDepth;
        int minSamplesSplit;
        int maxFeatures;
        bool bootstrap;
        int randomState;
        bool isFitted = false;
        int nFeatures = 0;
        std::vector<DecisionTree> trees;
        std::mt19937 internalRng;
        void buildTree(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
        std::vector<int> sampleBootstrap(int n);
        std::vector<int> sampleFeatures(int p, int maxFeatures);
        std::vector<std::vector<double>> predictAllTrees(const std::vector<std::vector<double>>& X);
        std::vector<double> aggregateMean(const std::vector<double>& preds);

    public:
        RandomForest(int Estimators = 100, int maxDepth = -1, int minSamplesSplit = 2, int maxFeatures = 0, bool bootstrap = true, int randomState = 0);
        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
        double predict(const std::vector<double>& X);
        std::vector<DecisionTree> getTrees() {return trees;};

};


#endif
