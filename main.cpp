#include <iostream>
#include "code/MLSuite/Dataset.h"
#include "code/MLSuite/LinRegModel.h"
#include <vector>
#include <Eigen/Dense>

int main() {
    try {
        // 1. Load the training data
        Dataset X_train("../data-preprocessing/data-files/regression/housing_data/housing_X_train_processed.csv", "train");
        Dataset y_train("../data-preprocessing/data-files/regression/housing_data/housing_y_train.csv", "train");

        // 2. Create and train the Linear Regression model
        LinRegModel model;
        model.fit(X_train, y_train, "L2");

        // 3. Print the learned parameters (theta)
        std::cout << "Learned parameters (theta):" << std::endl;
        std::cout << model.get_theta() << std::endl;
        std::cout << "---------------------------------" << std::endl;

        // 4. Load the test data
        Dataset X_test_dataset("../data-preprocessing/data-files/regression/housing_data/housing_X_test_processed.csv", "test");
        std::vector<float> test_data = X_test_dataset.get_data();
        int n_test_rows = test_data.size() / X_test_dataset.get_columns().size();
        int n_test_cols = X_test_dataset.get_columns().size();

        // 5. Convert test data to an Eigen Matrix
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_test(test_data.data(), n_test_rows, n_test_cols);

        // 6. Make predictions on the test data
        Eigen::VectorXf predictions = model.predict(X_test);

        // 7. Print the first few predictions
        std::cout << "Predictions on test data (first 10):" << std::endl;
        for(int i = 0; i < 10 && i < predictions.size(); ++i) {
            std::cout << predictions(i) << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
