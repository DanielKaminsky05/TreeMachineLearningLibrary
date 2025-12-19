#include "gtest/gtest.h"
#include "../code/MLSuite/DecisionTree.h"
#include <vector>

TEST(DecisionTreeTest, SimpleSplit) {
    // Test a simple decision tree logic
    // X = [[1], [2], [10], [11]]
    // Y = [0, 0, 1, 1]
    // Threshold ~ 5
    
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {10.0}, {11.0}};
    std::vector<double> Y = {0.0, 0.0, 1.0, 1.0};
    
    DecisionTree tree(5, 2); // MaxDepth 5, MinSplit 2
    tree.fit(X, Y);
    
    EXPECT_NEAR(tree.predict({1.5}), 0.0, 0.01);
    EXPECT_NEAR(tree.predict({10.5}), 1.0, 0.01);
}

TEST(DecisionTreeTest, EdgeCase_SingleSample) {
    std::vector<std::vector<double>> X = {{1.0}};
    std::vector<double> Y = {5.0};
    
    DecisionTree tree(5, 2);
    tree.fit(X, Y);
    
    EXPECT_NEAR(tree.predict({1.0}), 5.0, 0.01);
}
