cmake_minimum_required(VERSION 3.14)
project(MLSuiteTests)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(
    ${CMAKE_SOURCE_DIR}/../code/MLSuite
    ${CMAKE_SOURCE_DIR}/../3rdparty/eigen-5.0.0
)

# List of source files from the main project needed for linking
# Excluding main.cpp from the original project
set(MLSUITE_SOURCES
    ../code/MLSuite/Dataset.cpp
    ../code/MLSuite/LinRegModel.cpp
    ../code/MLSuite/LinearRegressionBuilder.cpp
    ../code/MLSuite/RandomForest.cpp
    ../code/MLSuite/RandomForestBuilder.cpp
    ../code/MLSuite/DecisionTree.cpp
    ../code/MLSuite/DecisionTreeBuilder.cpp
    ../code/MLSuite/RegressionBenchmark.cpp
    ../code/MLSuite/BenchmarkStrategy.cpp
    ../code/MLSuite/ClassicModelFactory.cpp
    ../code/MLSuite/XGBoostModel.cpp
    ../code/MLSuite/XGBoostBuilder.cpp
    ../code/MLSuite/LogRegModel.cpp
    ../code/MLSuite/LogisticRegressionBuilder.cpp
    ../code/MLSuite/LogRegModel.cpp
)

add_executable(runTests
    main.cpp
    TestRegressionBenchmark.cpp
    TestLinRegModel.cpp
    TestDecisionTree.cpp
    TestXGBoostModel.cpp
    TestRandomForest.cpp
    TestClassicModelFactory.cpp
    TestBuilders.cpp
    TestLogisticRegression.cpp
    MockModel.h
    ${MLSUITE_SOURCES}
)

target_link_libraries(runTests gtest gmock gtest_main)

include(GoogleTest)
gtest_discover_tests(runTests)
