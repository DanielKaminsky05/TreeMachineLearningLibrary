#ifndef RANDOMFORESTBUILDER_H
#define RANDOMFORESTBUILDER_H

#include "RandomForest.h"
#include <memory>

class RandomForestBuilder {
public:
    RandomForestBuilder();
    RandomForestBuilder& setEstimators(int estimators);
    RandomForestBuilder& setMaxDepth(int maxDepth);
    RandomForestBuilder& setMinSamplesSplit(int minSamplesSplit);
    RandomForestBuilder& setMaxFeatures(int maxFeatures);
    RandomForestBuilder& setBootstrap(bool bootstrap);
    RandomForestBuilder& setRandomState(int randomState);
    RandomForestBuilder& setIsClassification(bool isClassification); 

    std::unique_ptr<RandomForest> build();

private:
    int nEstimators;
    int mMaxDepth;
    int mMinSamplesSplit;
    int mMaxFeatures;
    bool mBootstrap;
    int mRandomState;
    bool mIsClassification; 
};
#endif // RANDOMFORESTBUILDER_H
