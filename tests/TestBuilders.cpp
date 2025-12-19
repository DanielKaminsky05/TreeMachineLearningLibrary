#include "gtest/gtest.h"
#include "../code/MLSuite/LinearRegressionBuilder.h"
#include "../code/MLSuite/DecisionTreeBuilder.h"
#include "../code/MLSuite/RandomForestBuilder.h"
#include "../code/MLSuite/XGBoostBuilder.h"
#include "../code/MLSuite/LinRegModel.h"
#include "../code/MLSuite/Dataset.h"
#include <fstream>
#include <cstdio>

// --- LinearRegressionBuilder Tests ---

class LinearRegressionBuilderTest : public ::testing::Test {
protected:
    std::string dummyFile = "builder_test_dummy.csv";

    void SetUp() override {
        std::ofstream ofs(dummyFile);
        ofs << "col1\n1.0\n2.0";
        ofs.close();
    }

    void TearDown() override {
        std::remove(dummyFile.c_str());
    }
};

TEST_F(LinearRegressionBuilderTest, BuildUnfitted) {
    LinearRegressionBuilder builder;
    auto model = builder.build_unfitted();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getName(), "Linear Regression");
}

TEST_F(LinearRegressionBuilderTest, FitWithData) {
    Dataset x(dummyFile, "train");
    Dataset y(dummyFile, "train");
    
    LinearRegressionBuilder builder;
    // Test that the unfitted model is built correctly.
    EXPECT_NO_THROW({
        LinRegModel model = builder.with_training_data(x, y)
                                   .with_regularization("L2")
                                   .with_lambda(0.5)
                                   .fit();
        EXPECT_EQ(model.getName(), "Linear Regression");
    });
}

// --- DecisionTreeBuilder Tests ---

TEST(DecisionTreeBuilderTest, BuildWithParams) {
    DecisionTreeBuilder builder;
    auto model = builder.setMaxDepth(5)
                        .setMinSamplesSplit(3)
                        .build();
    
    ASSERT_NE(model, nullptr);
    // DecisionTree class does not expose maxDepth or minSamplesSplit directly.
    // This test verifies the builder pattern mechanics.
}

// --- RandomForestBuilder Tests ---

TEST(RandomForestBuilderTest, BuildWithParams) {
    RandomForestBuilder builder;
    auto model = builder.setEstimators(20)
                        .setMaxDepth(4)
                        .setMinSamplesSplit(5)
                        .setMaxFeatures(2)
                        .setBootstrap(false)
                        .setRandomState(123)
                        .build();
    
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getName(), "Random Forest");
    // RandomForest exposes getTrees(), so we can check count after fitting.
    
    std::vector<std::vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> Y = {1.0, 2.0};
    model->fit(X, Y);
    EXPECT_EQ(model->getTrees().size(), 20);
}

// --- XGBoostBuilder Tests ---

TEST(XGBoostBuilderTest, BuildWithParams) {
    XGBoostBuilder builder;
    auto model = builder.setNEstimators(15)
                        .setLearningRate(0.05f)
                        .setMaxDepth(4)
                        .setSubsampleRatio(0.8f)
                        .setGamma(0.1f)
                        .setRegularization("L1")
                        .build();
    
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getName(), "XGBoost");
    
    // XGBoostModel exposes getters
    EXPECT_EQ(model->getNEstimators(), 15);
    EXPECT_FLOAT_EQ(model->getLearningRate(), 0.05f);
    EXPECT_EQ(model->getDepth(), 4);
    EXPECT_FLOAT_EQ(model->getSubsampleRatio(), 0.8f);
    EXPECT_FLOAT_EQ(model->getGamma(), 0.1f);
    EXPECT_EQ(model->getRegularization(), "L1");
}
