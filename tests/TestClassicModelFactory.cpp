#include "gtest/gtest.h"
#include "../code/MLSuite/ClassicModelFactory.h"
#include "../code/MLSuite/LinRegModel.h"
#include "../code/MLSuite/RandomForest.h"
#include "../code/MLSuite/XGBoostModel.h"
#include "../code/MLSuite/RegressionBenchmark.h"

class ClassicModelFactoryTest : public ::testing::Test {
protected:
    ClassicModelFactory factory;
};

TEST_F(ClassicModelFactoryTest, CreateLinReg) {
    auto model = factory.createLinRegModel();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getName(), "Linear Regression");
}

TEST_F(ClassicModelFactoryTest, CreateRandomForest) {
    auto model = factory.createRandomForestModel(10, 5, 2);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getName(), "Random Forest");
}

TEST_F(ClassicModelFactoryTest, CreateXGBoost) {
    auto model = factory.createXGBoostModel();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getName(), "XGBoost");
}

TEST_F(ClassicModelFactoryTest, RandomSearch_RandomForest) {
    // Mock data for random search
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Hyperparams: nEstimators, maxDepth, minSamplesSplit
    std::vector<std::vector<std::string>> params = {
        {"2", "5"},       // nEstimators
        {"2", "3"},       // maxDepth
        {"2"}             // minSamplesSplit
    };
    
    RegressionBenchmark benchmark;
    
    auto bestModel = factory.randomSearch("RandomForest", params, X, y, benchmark);
    ASSERT_NE(bestModel, nullptr);
    EXPECT_EQ(bestModel->getName(), "Random Forest");
}
