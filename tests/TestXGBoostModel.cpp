#include "gtest/gtest.h"
#include "../code/MLSuite/XGBoostModel.h"
#include "../code/MLSuite/XGBoostBuilder.h"
#include <vector>
#include <string>

class XGBoostModelTest : public ::testing::Test {
protected:
    std::vector<std::vector<double>> simpleX = {{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
    // Let's assume XOR-like or simple linear pattern. 
    // y = x1 + x2
    std::vector<double> simpleY = {2.0, 1.0, 1.0, 0.0};

    // Classification dataset
    std::vector<std::vector<double>> classificationX = {{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
    std::vector<double> classificationY = {1.0, 0.0, 0.0, 1.0}; // XOR
};

TEST_F(XGBoostModelTest, Initialization) {
    XGBoostModel xgb(10, 0.1f, 3, 1.0f, 0.0f, "L2");
    EXPECT_EQ(xgb.getName(), "XGBoost");
    EXPECT_EQ(xgb.getNEstimators(), 10);
    EXPECT_EQ(xgb.getLearningRate(), 0.1f);
}


TEST_F(XGBoostModelTest, IModelInterface_Predict) {
    XGBoostModel xgb(5, 0.5f, 2, 1.0f, 0.0f, "None");
    
    // Need to convert data for IModel interface
    std::vector<float> x_flat;
    std::vector<float> y_flat;
    for(size_t i=0; i<simpleX.size(); ++i) {
        x_flat.push_back((float)simpleX[i][0]);
        x_flat.push_back((float)simpleX[i][1]);
        y_flat.push_back((float)simpleY[i]);
    }
    
    std::vector<std::string> cols = {"f1", "f2"};
    
    xgb.fit(x_flat, cols, y_flat);
    
    std::vector<float> test_x = {1.0f, 0.0f};
    std::vector<float> preds = xgb.predict(test_x, cols);
    
    ASSERT_EQ(preds.size(), 1);
    EXPECT_NEAR(preds[0], 1.0f, 0.5);
}

TEST_F(XGBoostModelTest, FitAndPredict_Linear) {
    // Simple linear relationship y = 2*x1
    std::vector<std::vector<double>> linearX = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> linearY = {2.0, 4.0, 6.0, 8.0};

    XGBoostBuilder builder;
    builder.setNEstimators(10);
    builder.setLearningRate(0.3f);
    builder.setMaxDepth(1);
    builder.setRegularization("None");
    builder.setIsClassification(false);

    auto xgb = builder.build();
    xgb->fit(linearX, linearY);

    double pred1 = xgb->predict({1.0});
    double pred2 = xgb->predict({2.0});

    EXPECT_NEAR(pred1, 2.0, 0.5);
    EXPECT_NEAR(pred2, 4.0, 0.5);
}
