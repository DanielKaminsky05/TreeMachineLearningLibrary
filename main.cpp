#include <iostream>
#include "code/MLSuite/Dataset.h"
#include <vector>
#include <Eigen/Dense> // Include the necessary header

using namespace Eigen;

void perform_linear_regression() {
    // Eigen test
    std::cout << "--- Eigen Test Start ---" << std::endl;

    // Create a 2x2 matrix
    Matrix2f m;
    m << 1, 2,
         3, 4;

    // Create a 2x1 vector
    Vector2f v;
    v << 5, 6;

    // Perform matrix-vector multiplication
    Vector2f result = m * v;

    std::cout << "Matrix m:\n" << m << std::endl;
    std::cout << "\nVector v:\n" << v << std::endl;
    std::cout << "\nResult of m * v:\n" << result << std::endl;

    std::cout << "--- Eigen Test End ---" << std::endl;
}

int main() {
    perform_linear_regression(); // Run the Eigen test
    try {
        Dataset iris_train("../data-preprocessing/data-files/classification/iris_dataset/iris_X_train_processed.csv", "train");
        std::vector<float> data = iris_train.get_data();
        for(int i = 0; i < 10; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0; 
}
