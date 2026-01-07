cmake_minimum_required(VERSION 3.5)
project(Assignment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

include_directories(3rdparty/eigen-5.0.0)
include_directories(code)

add_executable(ui-demo
    	main.cpp
    	code/app/DemoRunner.cpp
    	code/app/DemoRunner.h
    	code/ui/MainWindow.cpp
    	code/ui/MainWindow.h
    	code/MLSuite/Dataset.cpp
    	code/MLSuite/LinRegModel.cpp
	code/MLSuite/LinearRegressionBuilder.cpp
	code/MLSuite/LogRegModel.cpp
	code/MLSuite/LogisticRegressionBuilder.cpp
	code/MLSuite/RandomForest.cpp
	code/MLSuite/RandomForestBuilder.cpp
    	code/MLSuite/DecisionTree.cpp
    	code/MLSuite/RegressionBenchmark.cpp
    	code/MLSuite/ClassificationBenchmark.cpp
    	code/MLSuite/BenchmarkStrategy.cpp
    	code/MLSuite/ClassicModelFactory.cpp
    	code/MLSuite/XGBoostModel.cpp
	code/MLSuite/XGBoostBuilder.cpp
)

find_package(Qt6 COMPONENTS Widgets QUIET)
if (Qt6_FOUND)
    target_link_libraries(ui-demo PRIVATE Qt6::Widgets)
else()
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    target_link_libraries(ui-demo PRIVATE Qt5::Widgets)
endif()

if (WIN32)
    target_link_libraries(ui-demo PRIVATE Psapi)
endif()
