#include <iostream>
#include <vector>
#include <memory>
// #include <stdexcept>

// Core components of the new design
#include "code/MLSuite/ClassicModelFactory.h"
#include "code/MLSuite/IModel.h"
#include "code/MLSuite/RegressionBenchmark.h"
#include "code/MLSuite/Dataset.h"

int main() {
    try {

        // 1. Create the factory and benchmark strategy objects
        ClassicModelFactory factory;
        RegressionBenchmark benchmark;
        
	Dataset x_train("../data-preprocessing/data-files/regression/housing_data/housing_X_train_processed.csv", "train");
	Dataset y_train("../data-preprocessing/data-files/regression/housing_data/housing_y_train.csv", "train");
	Dataset x_test("../data-preprocessing/data-files/regression/housing_data/housing_X_test_processed.csv", "test");
	Dataset y_test("../data-preprocessing/data-files/regression/housing_data/housing_y_test.csv", "test");


        // 3. Benchmark Linear Regression
        {
            std::cout << "\n--- Benchmarking Random Forest ---" << std::endl;
            // Create the model via the factory. It returns a std::unique_ptr<IModel>.
            std::unique_ptr<IModel> model = factory.createLinRegModel();

            // Fit the model using the IModel interface
            model->fit(x_train.get_data(), x_train.get_columns(), y_train.get_data());

            // Execute the benchmark. The benchmark works with any IModel.
            benchmark.execute(*model, x_test, y_test);
	
		std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
		for (int i = 0; i < 10; i++) {
			std::cout << results[i] << std::endl;

			}
        }

        // 4. Benchmark Random Forest
        {
            std::cout << "\n--- Benchmarking Random Forest ---" << std::endl;
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = factory.createRandomForestModel(50, 10, 2); // nEstimators, maxDepth, minSamplesSplit

            // Fit the model
            model->fit(x_train.get_data(), x_train.get_columns(), y_train.get_data());

            // Execute the benchmark. The benchmark works with any IModel.
            benchmark.execute(*model, x_test, y_test);

		std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
 		for (int i = 0; i < 10; i++) {
			std::cout << results[i] << std::endl;

		}           
        }

        std::cout << "\n--- Demo Complete ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
