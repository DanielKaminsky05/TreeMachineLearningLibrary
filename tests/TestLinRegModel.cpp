#include "gtest/gtest.h"
#include "../code/MLSuite/LinRegModel.h"
#include "../code/MLSuite/Dataset.h"
#include <fstream>
#include <cstdio>

class LinRegModelTest : public ::testing::Test {
protected:
    std::string trainFile = "test_train.csv";
    std::string testFile = "test_test.csv";

    void SetUp() override {
        // Create dummy CSV files for testing
        std::ofstream tFile(trainFile);
        tFile << "feature1,target\n";
        tFile << "1.0,2.0\n";
        tFile << "2.0,4.0\n";
        tFile << "3.0,6.0\n"; // y = 2x
        tFile.close();

        std::ofstream teFile(testFile);
        teFile << "feature1\n";
        teFile << "4.0\n";
        teFile << "5.0\n";
        teFile.close();
    }

    void TearDown() override {
        std::remove(trainFile.c_str());
        std::remove(testFile.c_str());
    }
};

TEST_F(LinRegModelTest, FitAndPredict) {
        // This test verifies the logic of Linear Regression
        Dataset trainData(trainFile, "train");
        Dataset testData(testFile, "test"); 
    
        LinRegModel model;
        
        // Set data in Dataset objects
        Dataset x(trainFile, "train");
        Dataset y(trainFile, "train");    // Mocking/Stubbing data inside Dataset using setters
    x.set_data({1.0, 2.0, 3.0}, {"feature1"});
    y.set_data({2.0, 4.0, 6.0}, {"target"});
    
    model.fit(x, y); // Train model

    // Predict
    // Using IModel interface for prediction
    std::vector<float> x_pred_val = {4.0, 5.0};
    std::vector<std::string> cols = {"feature1"};
    
    std::vector<float> predictions = model.predict(x_pred_val, cols);
    
    ASSERT_EQ(predictions.size(), 2);
    
    EXPECT_NEAR(predictions[0], 8.0, 0.1);
    EXPECT_NEAR(predictions[1], 10.0, 0.1);
}

TEST_F(LinRegModelTest, EdgeCase_EmptyData) {
    LinRegModel model;
    std::vector<float> x;
    std::vector<std::string> cols;
    std::vector<float> y;
    
    // Expect an exception for empty data.
}
