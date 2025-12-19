#include "gtest/gtest.h"
#include "../code/MLSuite/LogRegModel.h"
#include "../code/MLSuite/LogisticRegressionBuilder.h"
#include "../code/MLSuite/Dataset.h"
#include <fstream>
#include <cstdio>
#include <vector>
#include <string>

class LogisticRegressionTest : public ::testing::Test {
protected:
    std::string trainFile = "logistic_train.csv";
    std::string testFile = "logistic_test.csv";

    void SetUp() override {
        // Create dummy CSV files for a simple binary classification problem
        // Data: feature1, feature2, label
        // Roughly separable by feature1 + feature2 = constant
        std::ofstream tFile(trainFile);
        tFile << "feature1,feature2,label\n";
        tFile << "1.0,0.5,0\n";
        tFile << "1.5,0.8,0\n";
        tFile << "2.0,1.0,0\n";
        tFile << "0.5,1.5,1\n";
        tFile << "0.8,1.2,1\n";
        tFile << "2.5,1.2,0\n";
        tFile << "1.0,2.0,1\n";
        tFile.close();

        std::ofstream teFile(testFile);
        teFile << "feature1,feature2\n";
        teFile << "0.6,0.6\n"; // Expected class 0
        teFile << "2.0,2.0\n"; // Expected class 1
        teFile << "1.0,1.0\n"; // Expected class 0
        teFile << "1.5,1.5f\n"; // Expected class 1
        teFile.close();
    }

    void TearDown() override {
        std::remove(trainFile.c_str());
        std::remove(testFile.c_str());
    }
};

TEST_F(LogisticRegressionTest, FitAndPredict) {
    Dataset trainData(trainFile, "train");
    
    // Prepare X and y datasets
    std::vector<float> x_data;
    std::vector<float> y_data;
    std::vector<std::string> feature_cols = {"feature1", "feature2"};
    std::vector<std::string> label_col = {"label"};

    // Manually extract features and labels to create separate Dataset objects
    // For X_train
    x_data = {
        1.0f, 0.5f,
        1.5f, 0.8f,
        2.0f, 1.0f,
        0.5f, 1.5f,
        0.8f, 1.2f,
        2.5f, 1.2f,
        1.0f, 2.0f
    };
    Dataset X_train_ds(x_data, feature_cols);

    // For y_train
    y_data = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    Dataset y_train_ds(y_data, label_col);

    LogRegModel model = LogisticRegressionBuilder()
                                        .with_training_data(X_train_ds, y_train_ds)
                                        .with_learning_rate(0.1) // Use a slightly higher learning rate for faster convergence on simple data
                                        .with_num_iterations(5000) // Increase iterations
                                        .fit();

    // Test data for prediction
    std::vector<float> x_pred_values = {
        0.6f, 0.6f, // Expected class 0
        2.0f, 2.0f, // Expected class 1
        1.0f, 1.0f, // Expected class 0
        1.5f, 1.5f  // Expected class 1
    };

    std::vector<float> predictions = model.predict(x_pred_values, feature_cols);

    ASSERT_EQ(predictions.size(), 4);
    EXPECT_EQ(predictions[0], 0.0f);
    EXPECT_EQ(predictions[1], 1.0f);
    EXPECT_EQ(predictions[2], 0.0f);
    EXPECT_EQ(predictions[3], 1.0f);
}

TEST_F(LogisticRegressionTest, EdgeCase_EmptyData) {
    LogRegModel model;
    std::vector<float> x_empty;
    std::vector<std::string> cols_empty;
    std::vector<float> y_empty;
    
    // Test fit with empty data using IModel interface
    ASSERT_THROW(model.fit(x_empty, cols_empty, y_empty), std::invalid_argument);

    // Test predict with empty data using IModel interface
    // Note: predict() will return empty if input is empty, so no throw for predict directly.
    // However, if called after an unfitted model, it should throw.
    ASSERT_THROW(model.predict(x_empty, cols_empty), std::logic_error); // Model not fitted
}

TEST_F(LogisticRegressionTest, IModelInterfaceFitAndPredict) {
    LogRegModel model;

    // Data for IModel interface fit
    std::vector<float> x_fit_values = {
        1.0f, 0.5f,
        0.5f, 1.5f
    };
    std::vector<std::string> feature_cols = {"feature1", "feature2"};
    std::vector<float> y_fit_values = {0.0f, 1.0f};

    model.fit(x_fit_values, feature_cols, y_fit_values);

    // Test data for IModel interface predict
    std::vector<float> x_predict_values = {
        0.8f, 0.7f, // Expected 0
        0.7f, 0.8f  // Expected 1
    };
    
    std::vector<float> predictions = model.predict(x_predict_values, feature_cols);

    ASSERT_EQ(predictions.size(), 2);
    EXPECT_EQ(predictions[0], 0.0f);
    EXPECT_EQ(predictions[1], 1.0f);
}

// Optional: Test with L2 regularization
TEST_F(LogisticRegressionTest, FitAndPredictWithL2Regularization) {
    Dataset trainData(trainFile, "train");
    
    std::vector<float> x_data;
    std::vector<float> y_data;
    std::vector<std::string> feature_cols = {"feature1", "feature2"};
    std::vector<std::string> label_col = {"label"};

    x_data = {
        1.0f, 0.5f,
        1.5f, 0.8f,
        2.0f, 1.0f,
        0.5f, 1.5f,
        0.8f, 1.2f,
        2.5f, 1.2f,
        1.0f, 2.0f
    };
    Dataset X_train_ds(x_data, feature_cols);

    y_data = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    Dataset y_train_ds(y_data, label_col);

    LogRegModel model = LogisticRegressionBuilder()
                                        .with_training_data(X_train_ds, y_train_ds)
                                        .with_regularization("L2")
                                        .with_lambda(0.1) // Small lambda for L2
                                        .with_learning_rate(0.1) 
                                        .with_num_iterations(5000) 
                                        .fit();

    std::vector<float> x_pred_values = {
        0.6f, 0.6f, // Expected class 0
        2.0f, 2.0f, // Expected class 1
        1.0f, 1.0f, // Expected class 0
        1.5f, 1.5f  // Expected class 1
    };

    std::vector<float> predictions = model.predict(x_pred_values, feature_cols);

    ASSERT_EQ(predictions.size(), 4);
    EXPECT_EQ(predictions[0], 0.0f);
    EXPECT_EQ(predictions[1], 1.0f);
    EXPECT_EQ(predictions[2], 0.0f);
    EXPECT_EQ(predictions[3], 1.0f);
}
