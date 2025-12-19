// NOTE: this is a backup of main.cpp with Random Search and RF classif API.
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
            regressionBenchmark.execute(*model, x_test, y_test, fitMs);
    
        std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
        for (int i = 0; i < 10; i++) {
            std::cout << results[i] << std::endl;

			}
        }

        // 4. Benchmark Random Forest
        {
            std::cout << "\n--- Benchmarking Random Forest ---" << std::endl;
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = regressionFactory.createRandomForestModel(50, 10, 2, false); // nEstimators, maxDepth, minSamplesSplit, isClassif

            // Fit the model
            double fitMs = timeFit(*model, x_train, y_train);

            // Execute the benchmark. The benchmark works with any IModel.
            regressionBenchmark.execute(*model, x_test, y_test, fitMs);

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
            regressionBenchmark.execute(*model, x_test, y_test, fitMs);

        std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
         for (int i = 0; i < 10; i++) {
            std::cout << results[i] << std::endl;

        }           
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

            // Helper to reconstruct 2D double vectors from Dataset (1D float)
            // Ideally Dataset would expose this or RandomSearch would take Dataset.
            auto datasetToDouble2D = [](const Dataset& d) {
                const auto& data = d.get_data();
                const auto& cols = d.get_columns();
                size_t n_cols = cols.size();
                size_t n_rows = data.size() / n_cols;
                std::vector<std::vector<double>> out(n_rows, std::vector<double>(n_cols));
                for(size_t i=0; i<n_rows; ++i) {
                    for(size_t j=0; j<n_cols; ++j) {
                        out[i][j] = static_cast<double>(data[i*n_cols + j]);
                    }
                }
                return out;
            };
            
            auto datasetToDouble1D = [](const Dataset& d) {
                const auto& data = d.get_data();
                std::vector<double> out(data.size());
                for(size_t i=0; i<data.size(); ++i) out[i] = static_cast<double>(data[i]);
                return out;
            };

            // Perform Random Search using the Strategy (benchmark object)
            std::unique_ptr<IModel> bestModel = regressionFactory.randomSearch(
                "RandomForest",
                rfParams,
                datasetToDouble2D(x_train),
		datasetToDouble1D(y_train),
                regressionBenchmark // Passing the RegressionBenchmark strategy!
            );

            std::cout << "Best Random Forest Model found. Benchmarking it..." << std::endl;
            
            // Benchmark the best model found
            // Note: bestModel is already fitted by randomSearch on the full train set
            regressionBenchmark.execute(*bestModel, x_test, y_test, 0.0 /* fit time already spent */);
        }

        // 6. Classification benchmark (Random Forest on Iris dataset)
        {
            std::cout << "\n--- Classification Benchmark: Random Forest (Titanic) ---" << std::endl;
            // Pass true for isClassification
            std::unique_ptr<IModel> model = classificationFactory.createRandomForestModel(50, 10, 2, true); 

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
