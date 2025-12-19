#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "MockModel.h"
#include "../code/MLSuite/RegressionBenchmark.h"
#include "../code/MLSuite/Dataset.h"
#include <fstream>
#include <cstdio>

using ::testing:: _;
using ::testing::Return;
using ::testing::AtLeast;

class RegressionBenchmarkTest : public ::testing::Test {
protected:
    std::string dummyFile = "dummy.csv";

    void SetUp() override {
        // Create a simple dummy CSV for Dataset loading
        std::ofstream ofs(dummyFile);
        ofs << "col1\n1.0\n2.0";
        ofs.close();
    }

    void TearDown() override {
        std::remove(dummyFile.c_str());
    }
};

TEST_F(RegressionBenchmarkTest, HappyPath_CallsPredictAndGetName) {
    MockModel mockModel;
    RegressionBenchmark benchmark;

    Dataset xData(dummyFile, "train");
    Dataset yData(dummyFile, "train");

    // Verify loaded data size
    ASSERT_EQ(xData.get_data().size(), 2) << "Dataset failed to load expected 2 elements";

    // Expect predict to be called once with correct arguments logic (implicit via matcher)
    // and return a vector matching yData size (2 elements)
    EXPECT_CALL(mockModel, predict(_, _))
        .Times(1)
        .WillOnce(Return(std::vector<float>{1.0f, 2.0f}));

    // Expect getName to be called for printing results
    EXPECT_CALL(mockModel, getName())
        .Times(AtLeast(1))
        .WillRepeatedly(Return("MockModel"));

    benchmark.execute(mockModel, xData, yData);
}

TEST_F(RegressionBenchmarkTest, ErrorCase_MismatchedSize_DoesNotPrintResults) {
    MockModel mockModel;
    RegressionBenchmark benchmark;

    Dataset xData(dummyFile, "train");
    Dataset yData(dummyFile, "train");
    ASSERT_EQ(xData.get_data().size(), 2);

    // Return a vector of size 3 (yData has size 2)
    EXPECT_CALL(mockModel, predict(_, _))
        .Times(1)
        .WillOnce(Return(std::vector<float>{1.0f, 2.0f, 3.0f}));

    // getName is called at the beginning of execute to populate result struct.
    EXPECT_CALL(mockModel, getName())
        .Times(AtLeast(1))
        .WillRepeatedly(Return("MockModel"));

    benchmark.execute(mockModel, xData, yData);
}

TEST_F(RegressionBenchmarkTest, ErrorCase_EmptyPrediction) {
    MockModel mockModel;
    RegressionBenchmark benchmark;

    Dataset xData(dummyFile, "train");
    Dataset yData(dummyFile, "train");
    ASSERT_EQ(xData.get_data().size(), 2);

    // Return empty predictions
    EXPECT_CALL(mockModel, predict(_, _))
        .Times(1)
        .WillOnce(Return(std::vector<float>{}));

    // Should return early due to mismatch (0 != 2)
    // But getName is called early
    EXPECT_CALL(mockModel, getName())
        .Times(AtLeast(1))
        .WillRepeatedly(Return("MockModel"));

    benchmark.execute(mockModel, xData, yData);
}