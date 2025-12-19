#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "MockModel.h"
#include "../code/MLSuite/RegressionBenchmark.h"
#include "../code/MLSuite/Dataset.h"

using ::testing:: _;
using ::testing::Return;
using ::testing::AtLeast;

// Interaction Test using Mock Object
class RegressionBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(RegressionBenchmarkTest, ExecuteCallsFitAndPredict) {
    MockModel mockModel;
    RegressionBenchmark benchmark;

    // RegressionBenchmark::execute calls model.predict but not model.fit.
    // The Dataset objects are configured to have 2 data points for simple testing.
    
    std::string dummyFile = "dummy.csv";
    std::ofstream ofs(dummyFile);
    ofs << "target\n1.0\n2.0";
    ofs.close();

    Dataset xData(dummyFile, "train"); // type doesn't matter much for this test
    Dataset yData(dummyFile, "train");

    EXPECT_CALL(mockModel, predict(_, _))
        .Times(::testing::AtLeast(1))
        .WillRepeatedly(Return(std::vector<float>{1.0f, 2.0f})); // Return dummy predictions matching size

    EXPECT_CALL(mockModel, getName())
        .WillRepeatedly(Return("MockModel"));

    benchmark.execute(mockModel, xData, yData);

    EXPECT_CALL(mockModel, getName())
        .WillRepeatedly(Return("MockModel"));

    benchmark.execute(mockModel, xData, yData);
    
    // Clean up
    std::remove(dummyFile.c_str());
}
