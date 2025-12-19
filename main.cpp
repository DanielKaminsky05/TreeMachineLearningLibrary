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
            "../data-preprocessing/data-files/classification/iris_dataset/iris_X_train_processed.csv",
            "../data-preprocessing/data-files/classification/iris_dataset/iris_y_train.csv",
            "../data-preprocessing/data-files/classification/iris_dataset/iris_X_test_processed.csv",
            "../data-preprocessing/data-files/classification/iris_dataset/iris_y_test.csv");
        RegressionBenchmark benchmark;
        ClassificationBenchmark classificationBenchmark;
        
        Dataset x_train = regressionFactory.loadTrainFeatures();
        Dataset y_train = regressionFactory.loadTrainTargets();
        Dataset x_test = regressionFactory.loadTestFeatures();
        Dataset y_test = regressionFactory.loadTestTargets();

        // Classification datasets (Iris)
        Dataset cx_train = classificationFactory.loadTrainFeatures();
        Dataset cy_train = classificationFactory.loadTrainTargets();
        Dataset cx_test = classificationFactory.loadTestFeatures();
        Dataset cy_test = classificationFactory.loadTestTargets();

        auto timeFit = [](IModel& model, const Dataset& features, const Dataset& targets) -> double {
            auto start = std::chrono::high_resolution_clock::now();
            model.fit(features.get_data(), features.get_columns(), targets.get_data());
            auto end = std::chrono::high_resolution_clock::now();
            return millisBetween(start, end);
        };


        // 3. Benchmark Linear Regression
        {
            std::cout << "\n--- Benchmarking Linear Regression ---" << std::endl;
            // Create the model via the factory. It returns a std::unique_ptr<IModel>.
            std::unique_ptr<IModel> model = regressionFactory.createLinRegModel();

            // Fit the model using the IModel interface
            double fitMs = timeFit(*model, x_train, y_train);

            // Execute the benchmark. The benchmark works with any IModel.
            benchmark.execute(*model, x_test, y_test, fitMs);
    
        std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
        for (int i = 0; i < 10; i++) {
            std::cout << results[i] << std::endl;

			}
        }

        // 4. Benchmark Random Forest
        {
            std::cout << "\n--- Benchmarking Random Forest ---" << std::endl;
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = regressionFactory.createRandomForestModel(50, 10, 2); // nEstimators, maxDepth, minSamplesSplit

            // Fit the model
            double fitMs = timeFit(*model, x_train, y_train);

            // Execute the benchmark. The benchmark works with any IModel.
            benchmark.execute(*model, x_test, y_test, fitMs);

        std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
         for (int i = 0; i < 10; i++) {
            std::cout << results[i] << std::endl;

		}           
        }

        // 5. Benchmark XGBoost
        {
            std::cout << "\n--- Benchmarking XGBoost ---" << std::endl;
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = regressionFactory.createXGBoostModel(50, 0.1f, 10, 0.8f, 0.1f, "L2");

            // Fit the model
            double fitMs = timeFit(*model, x_train, y_train);

            // Execute the benchmark. The benchmark works with any IModel.
            benchmark.execute(*model, x_test, y_test, fitMs);

        std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
         for (int i = 0; i < 10; i++) {
            std::cout << results[i] << std::endl;

        }           
        }

        // 6. Classification benchmark (Random Forest on Iris dataset)
        {
            std::cout << "\n--- Classification Benchmark: Random Forest (Iris) ---" << std::endl;
            std::unique_ptr<IModel> model = classificationFactory.createRandomForestModel(50, 10, 2);
            double fitMs = timeFit(*model, cx_train, cy_train);
            classificationBenchmark.execute(*model, cx_test, cy_test, fitMs);
        }

        std::cout << "\n--- Demo Complete ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
