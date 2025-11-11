#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>

// Core components of the new design
#include "code/MLSuite/ClassicModelFactory.h"
#include "code/MLSuite/IModel.h"
#include "code/MLSuite/RegressionBenchmark.h"
#include "code/MLSuite/Dataset.h"

// --- Helper function to create some dummy data ---
void createDummyData(std::vector<std::vector<float>>& features, std::vector<float>& targets, int num_samples, int num_features) {
    features.clear();
    targets.clear();
    features.resize(num_samples, std::vector<float>(num_features));
    targets.resize(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            // Simple, predictable data
            features[i][j] = static_cast<float>(i + j);
        }
        // Create a simple linear relationship for the target: y = 2*f1 + 3.5*f2 + 5
        targets[i] = 2.0f * features[i][0] + 3.5f * (features[i].size() > 1 ? features[i][1] : 0.0f) + 5.0f;
    }
}


int main() {
    try {
        std::cout << "--- Machine Learning Benchmark Demo ---" << std::endl;

        // 1. Create the factory and benchmark strategy objects
        ClassicModelFactory factory;
        RegressionBenchmark benchmark;

        // 2. Create some dummy data for training and testing
        std::vector<std::vector<float>> train_features, test_features;
        std::vector<float> train_targets, test_targets;
        createDummyData(train_features, train_targets, 100, 2);
        createDummyData(test_features, test_targets, 20, 2);
        
        // Create a Dataset object for the benchmark using our new in-memory constructor
        Dataset test_dataset(test_features, test_targets);


        // 3. Benchmark Linear Regression
        {
            std::cout << "\n--- Benchmarking Linear Regression ---" << std::endl;
            // Create the model via the factory. It returns a std::unique_ptr<IModel>.
            std::unique_ptr<IModel> model = factory.createLinRegModel();

            // Fit the model using the IModel interface
            model->fit(train_features, train_targets);

            // Execute the benchmark. The benchmark works with any IModel.
            benchmark.execute(*model, test_dataset);
        }

        // 4. Benchmark Random Forest
        {
            std::cout << "\n--- Benchmarking Random Forest ---" << std::endl;
            // Create a different model from the same factory
            std::unique_ptr<IModel> model = factory.createRandomForestModel(50, 10, 2); // nEstimators, maxDepth, minSamplesSplit

            // Fit the model
            model->fit(train_features, train_targets);

            // Use the *exact same* benchmark object on the new model
            benchmark.execute(*model, test_dataset);
        }

        std::cout << "\n--- Demo Complete ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
