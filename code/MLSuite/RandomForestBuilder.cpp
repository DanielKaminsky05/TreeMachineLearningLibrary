#include "RandomForestBuilder.h"

RandomForestBuilder::RandomForestBuilder()
    : nEstimators(100),
      mMaxDepth(-1),
      mMinSamplesSplit(2),
      mMaxFeatures(0),
      mBootstrap(true),
      mRandomState(0),
      mIsClassification(false) {} 

RandomForestBuilder& RandomForestBuilder::setEstimators(int estimators) {
	nEstimators = estimators;
	return *this;
}

RandomForestBuilder& RandomForestBuilder::setMaxDepth(int maxDepth) {
    	mMaxDepth = maxDepth;
    	return *this;
}

RandomForestBuilder& RandomForestBuilder::setMinSamplesSplit(int minSamplesSplit) {
    	mMinSamplesSplit = minSamplesSplit;
    	return *this;
}

RandomForestBuilder& RandomForestBuilder::setMaxFeatures(int maxFeatures) {
    	mMaxFeatures = maxFeatures;
    	return *this;
}

RandomForestBuilder& RandomForestBuilder::setBootstrap(bool bootstrap) {
    	mBootstrap = bootstrap;
    	return *this;
}

RandomForestBuilder& RandomForestBuilder::setRandomState(int randomState) {
    	mRandomState = randomState;
    	return *this;
}

RandomForestBuilder& RandomForestBuilder::setIsClassification(bool isClassification) {
    	mIsClassification = isClassification;
    	return *this;
}

std::unique_ptr<RandomForest> RandomForestBuilder::build() {
    	return std::make_unique<RandomForest>(nEstimators, mMaxDepth, mMinSamplesSplit, mMaxFeatures, mBootstrap, mRandomState, mIsClassification);
}
