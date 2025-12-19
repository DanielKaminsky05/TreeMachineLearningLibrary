// NOTE: This is a backup of the main file with RF classif, flexible benchmark strategies, and fixed print statements + type conversions
// added Logreg.

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
// #include <stdexcept>

// Core components of the new design
#include "code/MLSuite/ClassicModelFactory.h"
#include "code/MLSuite/IModel.h"
#include "code/MLSuite/RegressionBenchmark.h"
#include "code/MLSuite/ClassificationBenchmark.h"
#include "code/MLSuite/Dataset.h"

int main() {
    try {

        // 1. Create the factory and benchmark strategy objects
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

        // 3. Benchmark Linear Regression
        {
            // Create the model via the factory. It returns a std::unique_ptr<IModel>.
            std::unique_ptr<IModel> model = regressionFactory.createLinRegModel();

            // Train and Execute benchmark
            regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);
        }

        // 4. Benchmark Random Forest
        {
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = regressionFactory.createRandomForestModel(50, 10, 2, false); // nEstimators, maxDepth, minSamplesSplit, isClassif

            // Train and Execute benchmark
            regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);      
        }


        // 5. Benchmark XGBoost
        {
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = regressionFactory.createXGBoostModel(50, 0.1f, 10, 0.8f, 0.1f, "L2");

            // Train and Execute benchmark
            regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);        
        }

        // 5.5 Random Search Demo (Strategy Pattern)
        {
            std::cout << "\n--- Hyperparameter Search: Random Forest (Regression) ---" << std::endl;
            
            // Define Hyperparameter Grid
            std::vector<std::vector<std::string>> rfParams = {
                {"10", "50", "100"}, // nEstimators
                {"5", "10", "20"},   // maxDepth
                {"2", "5", "10"}     // minSamplesSplit
            };

            // Perform Random Search using the Strategy (benchmark object)
            std::unique_ptr<IModel> bestModel = regressionFactory.randomSearch(
                "RandomForest",
                rfParams,
                x_train.get_data_as_double_2d(),
				y_train.get_data_as_double_1d(),
                regressionBenchmark // Passing the RegressionBenchmark strategy!
            );

            std::cout << "Best Random Forest Model found. Benchmarking it..." << std::endl;
            
            // Benchmark the best model found
            // Note: bestModel is already fitted by randomSearch on the full train set
            regressionBenchmark.execute(*bestModel, x_test, y_test, 0.0 /* fit time already spent */);
        }

        // 6. Classification benchmark (Random Forest on Iris dataset)
        {
            // Pass true for isClassification
            std::unique_ptr<IModel> model = classificationFactory.createRandomForestModel(50, 10, 2, true); 

            // Train and Execute benchmark
            classificationBenchmark.trainAndExecute(*model, cx_train, cy_train, cx_test, cy_test);
        }

        // NOTE: XGB classif benchmark
        {
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = regressionFactory.createXGBoostModel(50, 0.1, 3, 1.0, 0.0, "L2", true); 

            // Train and Execute benchmark
            classificationBenchmark.trainAndExecute(*model, cx_train, cy_train, cx_test, cy_test);      
        }

	{ // NOTE: log reg model test 
		std::unique_ptr<IModel> model = classificationFactory.createLogRegModel();

		classificationBenchmark.trainAndExecute(*model, cx_train, cy_train, cx_test, cy_test);
		
	}

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

