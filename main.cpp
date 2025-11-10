#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "code/MLSuite/Dataset.h"
#include "code/MLSuite/LinRegModel.h"
#include "code/MLSuite/LinearRegressionBuilder.h"

int main() {
    try {
        // 1. Load all necessary data
        Dataset X_train("../data-preprocessing/data-files/regression/housing_data/housing_X_train_processed.csv", "train");
        Dataset y_train("../data-preprocessing/data-files/regression/housing_data/housing_y_train.csv", "train");
        Dataset X_test_dataset("../data-preprocessing/data-files/regression/housing_data/housing_X_test_processed.csv", "test");
        // y_test is loaded but not used in this example, but would be needed for evaluation 
        // Dataset y_test("../data-preprocessing/data-files/regression/housing_data/housing_y_test.csv", "test");

        // 2. Create and train the model using the builder
        LinearRegressionBuilder builder;
        LinRegModel model = builder.with_training_data(X_train, y_train).with_regularization("L2").with_lambda(0.5).fit();

        // 3. Print the learned parameters (theta)
        std::cout << "Learned parameters (theta):" << std::endl;
        std::cout << model.get_theta() << std::endl;

        // 4. Prepare the test data for prediction
        std::vector<float> test_data = X_test_dataset.get_data();
        int n_test_rows = test_data.size() / X_test_dataset.get_columns().size();
        int n_test_cols = X_test_dataset.get_columns().size();

        // Create an Eigen Matrix view of the test data
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_test(test_data.data(), n_test_rows, n_test_cols);

        // 5. Make predictions
        Eigen::VectorXf predictions = model.predict(X_test);

        // 6. Print the first few predictions
        std::cout << "Predictions on test data (first 10):" << std::endl;
        for (int i = 0; i < 10 && i < predictions.size(); ++i) {
            std::cout << predictions(i) << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
