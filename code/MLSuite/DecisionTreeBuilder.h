#ifndef DECISIONTREEBUILDER_H
#define DECISIONTREEBUILDER_H

#include "DecisionTree.h"
#include <memory>

class DecisionTreeBuilder {
public:
    DecisionTreeBuilder();
    DecisionTreeBuilder& setMaxDepth(int maxDepth);
    DecisionTreeBuilder& setMinSamplesSplit(int minSamplesSplit);
    DecisionTreeBuilder& setIsClassification(bool isClassification);

    std::unique_ptr<DecisionTree> build();

private:
    int mMaxDepth;
    int mMinSamplesSplit;
    bool mIsClassification;
};

#endif // DECISIONTREEBUILDER_H
