#include "DecisionTreeBuilder.h"
#include <stdexcept> // For std::invalid_argument

DecisionTreeBuilder::DecisionTreeBuilder() 
    : mMaxDepth(10), mMinSamplesSplit(2) {}

DecisionTreeBuilder& DecisionTreeBuilder::setMaxDepth(int maxDepth) {
    if (maxDepth <= 0) {
        throw std::invalid_argument("maxDepth must be a positive integer.");
    }
    mMaxDepth = maxDepth;
    return *this;
}

DecisionTreeBuilder& DecisionTreeBuilder::setMinSamplesSplit(int minSamplesSplit) {
    if (minSamplesSplit < 2) {
        throw std::invalid_argument("minSamplesSplit must be at least 2.");
    }
    mMinSamplesSplit = minSamplesSplit;
    return *this;
}

std::unique_ptr<DecisionTree> DecisionTreeBuilder::build() {
    return std::make_unique<DecisionTree>(mMaxDepth, mMinSamplesSplit);
}
