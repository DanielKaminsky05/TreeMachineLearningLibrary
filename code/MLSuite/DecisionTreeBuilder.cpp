#include "DecisionTreeBuilder.h"

DecisionTreeBuilder::DecisionTreeBuilder() 
    : mMaxDepth(10), mMinSamplesSplit(2) {}

DecisionTreeBuilder& DecisionTreeBuilder::setMaxDepth(int maxDepth) {
    mMaxDepth = maxDepth;
    return *this;
}

DecisionTreeBuilder& DecisionTreeBuilder::setMinSamplesSplit(int minSamplesSplit) {
    mMinSamplesSplit = minSamplesSplit;
    return *this;
}

std::unique_ptr<DecisionTree> DecisionTreeBuilder::build() {
    return std::make_unique<DecisionTree>(mMaxDepth, mMinSamplesSplit);
}
