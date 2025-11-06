#include <iostream>
#include "code/MLSuite/Dataset.h"
#include <vector>

int main() {
    try {
        Dataset iris_train("data-preprocessing/data-files/classification/iris_dataset/iris_X_train_processed.csv", "train");
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
