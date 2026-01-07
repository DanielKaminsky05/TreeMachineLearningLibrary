#include "DemoRunner.h"

#include <chrono>
#include <iomanip>
#include <iostream> // Keep for stringstream
#include <memory>
#include <sstream>
#include <vector>

// Core components of the new design
#include "MLSuite/BenchmarkStrategy.h"
#include "MLSuite/ClassicModelFactory.h"
#include "MLSuite/IModel.h"
#include "MLSuite/RegressionBenchmark.h"
#include "MLSuite/ClassificationBenchmark.h"
#include "MLSuite/Dataset.h"

namespace {
void logLine(const DemoRunner::LogFn& log, const std::string& line) {
	if (log) {
		log(line);
    	} else {
        	std::cout << line << std::endl; // Fallback to cout if log function is not provided
    	}
}

std::string formatFloat(float value) {
	std::ostringstream oss;
    	oss << std::fixed << std::setprecision(6) << value;
    	return oss.str();
}

} // namespace

int DemoRunner::runFullDemo(const LogFn& log) {
	try {
        	ClassicModelFactory regressionFactory(
            		"../data-preprocessing/data-files/regression/housing_data/housing_X_train_processed.csv",
            		"../data-preprocessing/data-files/regression/housing_data/housing_y_train.csv",
            		"../data-preprocessing/data-files/regression/housing_data/housing_X_test_processed.csv",
            		"../data-preprocessing/data-files/regression/housing_data/housing_y_test.csv");
        	ClassicModelFactory classificationFactory(
            		"../data-preprocessing/data-files/classification/titanic_dataset/titanic_X_train_processed.csv",
            		"../data-preprocessing/data-files/classification/titanic_dataset/titanic_y_train.csv",
            		"../data-preprocessing/data-files/classification/titanic_dataset/titanic_X_test_processed.csv",
            		"../data-preprocessing/data-files/classification/titanic_dataset/titanic_y_test.csv");

        	RegressionBenchmark regressionBenchmark;
        	ClassificationBenchmark classificationBenchmark;

        	Dataset x_train = regressionFactory.loadTrainFeatures();
        	Dataset y_train = regressionFactory.loadTrainTargets();
        	Dataset x_test = regressionFactory.loadTestFeatures();
        	Dataset y_test = regressionFactory.loadTestTargets();

        	// Classification datasets (Titanic)
        	Dataset cx_train = classificationFactory.loadTrainFeatures();
        	Dataset cy_train = classificationFactory.loadTrainTargets();
        	Dataset cx_test = classificationFactory.loadTestFeatures();
        	Dataset cy_test = classificationFactory.loadTestTargets();

        	{
            		logLine(log, "--- Benchmarking Linear Regression ---");
            		std::unique_ptr<IModel> model = regressionFactory.createLinRegModel();

            		regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);
        	}

        	{
            		logLine(log, "--- Benchmarking Random Forest ---");
            		std::unique_ptr<IModel> model = regressionFactory.createRandomForestModel(50, 10, 2, false); // nEstimators, maxDepth, minSamplesSplit, isClassif

            		// Train and Execute benchmark
            		regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);
        	}

        	{
            		logLine(log, "--- Benchmarking XGBoost ---");
            		// Create a different model from the same factory
            		std::unique_ptr<IModel> model = regressionFactory.createXGBoostModel(50, 0.1f, 10, 0.8f, 0.1f, "L2");

            		// Train and Execute benchmark
            		regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);
        	}

        	{
            		logLine(log, "\n--- Hyperparameter Search: Random Forest (Regression) ---");

            		std::vector<std::vector<std::string>> rfParams = {
                		{"10", "50", "100"}, // nEstimators
                		{"5", "10", "20"},   // maxDepth
                		{"2", "5", "10"}     // minSamplesSplit
            	};

            	std::unique_ptr<IModel> bestModel = regressionFactory.randomSearch(
                	"RandomForest",
                	rfParams,
                	x_train.get_data_as_double_2d(),
                	y_train.get_data_as_double_1d(),
                	regressionBenchmark, // Passing the RegressionBenchmark strategy!
                	log
            	);

            	logLine(log, "Best Random Forest Model found. Benchmarking it...");

            	regressionBenchmark.execute(*bestModel, x_test, y_test, 0.0 /* fit time already spent */);
        }

        {
            	logLine(log, "--- Classification Benchmark: Random Forest (Titanic) ---");
            	// Pass true for isClassification
            	std::unique_ptr<IModel> model = classificationFactory.createRandomForestModel(50, 10, 2, true);

            	// Train and Execute benchmark
            	classificationBenchmark.trainAndExecute(*model, cx_train, cy_train, cx_test, cy_test);
        }

        // NOTE: XGB classif benchmark
        {
            	logLine(log, "--- Classification Benchmark: XGBoost (Titanic) ---");
            	// Create a different model from the same factory
            	std::unique_ptr<IModel> model = regressionFactory.createXGBoostModel(50, 0.1, 3, 1.0, 0.0, "L2", true);

            	// Train and Execute benchmark
            	classificationBenchmark.trainAndExecute(*model, cx_train, cy_train, cx_test, cy_test);
        }
		
	// NOTE: log reg model test
        {             	
		logLine(log, "--- Classification Benchmark: Logistic Regression (Titanic) ---");
            	std::unique_ptr<IModel> model = classificationFactory.createLogRegModel();

            classificationBenchmark.trainAndExecute(*model, cx_train, cy_train, cx_test, cy_test);
        }

        logLine(log, "--- Demo Complete ---");

        return 0;

    } catch (const std::exception& e) {
        	logLine(log, std::string("An error occurred: ") + e.what());

        return 1;
    }
}
