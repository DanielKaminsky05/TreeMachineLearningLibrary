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
    std::string trainFileX = "logistic_train_x.csv";
    std::string trainFileY = "logistic_train_y.csv";
    std::string testFile = "logistic_test.csv";

    void SetUp() override {
        // Create dummy CSV files for a simple binary classification problem
        // Data: feature1, feature2, label
        // Roughly separable by feature1 + feature2 = constant
        std::ofstream tFileX(trainFileX);
        tFileX << "feature1,feature2\n";
        tFileX << "1.0,0.5\n";
        tFileX << "1.5,0.8\n";
        tFileX << "2.0,1.0\n";
        tFileX << "0.5,1.5\n";
        tFileX << "0.8,1.2\n";
        tFileX << "2.5,1.2\n";
        tFileX << "1.0,2.0\n";
        tFileX.close();

        std::ofstream tFileY(trainFileY);
        tFileY << "label\n";
        tFileY << "0\n";
        tFileY << "0\n";
        tFileY << "0\n";
        tFileY << "1\n";
        tFileY << "1\n";
        tFileY << "0\n";
        tFileY << "1\n";
        tFileY.close();

        std::ofstream teFile(testFile);
        teFile << "feature1,feature2\n";
        teFile << "0.6,0.6\n"; // Expected class 0
        teFile << "2.0,2.0\n"; // Expected class 1
        teFile << "1.0,1.0\n"; // Expected class 0
        teFile << "1.5,1.5f\n"; // Expected class 1
        teFile.close();
    }

    void TearDown() override {
        std::remove(trainFileX.c_str());
        std::remove(trainFileY.c_str());
        std::remove(testFile.c_str());
    }
};



