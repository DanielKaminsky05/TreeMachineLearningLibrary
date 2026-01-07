#include "gtest/gtest.h"
#include "../code/MLSuite/RandomForest.h"

class RandomForestTest : public ::testing::Test {
protected:
    std::vector<std::vector<double>> simpleX = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> simpleY = {10.0, 20.0, 30.0, 40.0, 50.0}; // y = 10x
};

TEST_F(RandomForestTest, FitAndPredict_Simple) {
    // 5 estimators, depth 3
    RandomForest rf(10, 3, 2, 1, true, 42); 
    
    rf.fit(simpleX, simpleY);
    
    // Test on seen data
    double pred = rf.predict({3.0});
    EXPECT_NEAR(pred, 30.0, 5.0); // Random Forest average prediction might be slightly off
    
    // Test on unseen data (interpolation)
    double pred2 = rf.predict({2.5});
    // Predictions from decision trees can be step-wise, so a range check is appropriate.
    EXPECT_GT(pred2, 15.0);
    EXPECT_LT(pred2, 35.0);
}

TEST_F(RandomForestTest, FeatureCheck) {
    RandomForest rf(10, 3, 2, 1, true, 42); 
    rf.fit(simpleX, simpleY);
    
    // Verify that trees were built (RandomForest::fit() builds them).
    EXPECT_FALSE(rf.getTrees().empty());
    EXPECT_EQ(rf.getTrees().size(), 10);
}
