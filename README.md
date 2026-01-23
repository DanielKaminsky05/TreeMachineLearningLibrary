# 3307 Final OOP Deliverable

This README details the final submission for our course project for OOP design patterns, where we implemented traditional machine learning libraries in C++ from scratch with our custom Dataset implementation. Users can load CSV data, train the model, and benchmark the results and performance of the model using our abstracted library.

[Demonstration Video Link](https://drive.google.com/file/d/1hTJA2yeSf_bWunvn1wNWdFWe47EQi0Iw/view?usp=sharing)

# Repo structure 

```
.
├── 3rdparty # library used for linear algebra.
│   ├── eigen-5.0.0
│   └── eigen-5.0.0.zip
├── code # source code 
│   ├── app # backend logic for QT Widgets, calls MLSuite classes.
│   │   ├── DemoRunner.cpp
│   │   └── DemoRunner.h 
│   ├── MLSuite # core machine learning logic.
│   │   ├── .gitignore 
│   │   ├── BenchmarkStrategy.cpp
│   │   ├── BenchmarkStrategy.h
│   │   ├── ClassicModelFactory.cpp
│   │   ├── ClassicModelFactory.h 
│   │   ├── ClassificationBenchmark.cpp
│   │   ├── ClassificationBenchmark.h
│   │   ├── Dataset.cpp
│   │   ├── Dataset.h
│   │   ├── DecisionTree.cpp
│   │   ├── DecisionTree.h
│   │   ├── DecisionTreeBuilder.cpp
│   │   ├── DecisionTreeBuilder.h
│   │   ├── HyperparameterSearch.cpp
│   │   ├── HyperparameterSearch.h
│   │   ├── IModel.h
│   │   ├── LinearRegressionBuilder.cpp
│   │   ├── LinearRegressionBuilder.h
│   │   ├── LinRegModel.cpp
│   │   ├── LinRegModel.h 
│   │   ├── LogisticRegressionBuilder.cpp
│   │   ├── LogisticRegressionBuilder.h
│   │   ├── LogRegModel.cpp
│   │   ├── LogRegModel.h
│   │   ├── main.cpp
│   │   ├── ProjectTemplate.pro
│   │   ├── RandomForest.cpp
│   │   ├── RandomForest.h
│   │   ├── RandomForestBuilder.cpp
│   │   ├── RandomForestBuilder.h
│   │   ├── RegressionBenchmark.cpp
│   │   ├── RegressionBenchmark.h
│   │   ├── XGBoostBuilder.cpp
│   │   ├── XGBoostBuilder.h
│   │   ├── XGBoostModel.cpp
│   │   └── XGBoostModel.h
│   └── ui # UI code with QT Widgets 
│       ├── MainWindow.cpp
│       └── MainWindow.h 
├── data-preprocessing # data preprocessing code and files for datasets used.
│   ├── data-files
│   ├── preprocessing-code
│   ├── source-files
│   └── README.md
│   ├── 3307 Deliverable #2 - Henrique & Daniel.pdf 
│   ├── 3307 Deliverable #2 - Thomson Lam.pdf
│   ├── 3307 Deliverable #3 - Henrique & Daniel.pdf 
│   └── 3307 Deliverable #3 - Thomson Lam.pdf
│   └── README.md # please find more details here.
├── tests # directory for unit testing.
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── MockModel.h
│   ├── TestBuilders.cpp
│   ├── TestClassicModelFactory.cpp
│   ├── TestDecisionTree.cpp
│   ├── TestLinRegModel.cpp
│   ├── TestLogisticRegression.cpp
│   ├── TestRandomForest.cpp
│   ├── TestRegressionBenchmark.cpp
│   └── TestXGBoostModel.cpp
├── .gitignore
├── CMakeLists.txt
├── demo.cpp # example demonstration file for using MLSuite without the QT Widgets UI.
├── main.cpp # entry point of the project's compiled executable.
└── README.md 
```


## About 


# Course Project: A Classic Machine Learning Suite library Built with C++

## Introduction

This repository contains code that implements a high-fidelity prototype for classical machine learning methods such as linear regression, random forest and XGBoost with limited customizability using several design patterns such as the Strategy, Builder and Factory patterns to ensure loose coupling and high cohesiveness for implementing and running the algorithms on a diverse set of datasets loaded through our own implementation of a Dataset class.

While we initially set our sights on making a machine learning and Neural Random Forest library after taking inspiration from the [Neural Random Forests Paper](https://arxiv.org/abs/1604.07143), due to the complexity and stochastic nature of neural networks, we were unable to produce a high-fidelity, easy-to-use library for CPU-based deep learning with tabular datasets. As a result, we pivoted our focus to polishing existing classes and improving the performance and efficiency of traditional ML classes instead. One of the key changes added was hyperparameter optimization; we added Random Search to our API due to its efficiency.

## Prerequisites

Before you begin, ensure you have the following installed:
- C++17 compliant compiler (e.g., GCC, Clang, MSVC)
- CMake (version 3.14 or later)
- Qt (version 5 or 6)

## Building and Running

### Using QT Creator

1.  Open the `CMakeLists.txt` file in the project root with Qt Creator.
2.  Configure the project by selecting a kit with a C++17 compiler.
3.  Make sure that `3rdparty` is in the project root and is included in CMakeLists.txt:
    ```cmake
    include_directories(3rdparty/eigen-5.0.0)
    ```
4.  Build and run the `assignment2` executable target.

### Using the Terminal

1.  Create a build directory:
    ```sh
    mkdir build && cd build
    ```
2.  Run CMake and build the project:
    ```sh
    cmake ..
    make
    ```
3.  Run the executables:
    ```sh
    ./demo
    ./ui-demo
    ```
    You can also refer to the `demo.cpp` for more usage examples.

## Running Tests

The project uses GoogleTest for unit testing. To run the tests:

1.  Navigate to the build directory:
    ```sh
    cd build
    ```
2.  Build the test executable:
    ```sh
    make runTests
    ```
3.  Run the tests:
    ```sh
    ./runTests
    ```

## Example Usage

Here is a basic example of how to use the machine learning suite (from `demo.cpp`):

```cpp
#include <iostream>
#include <vector>
#include <memory>

// Core components of the new design
#include "code/MLSuite/ClassicModelFactory.h"
#include "code/MLSuite/IModel.h"
#include "code/MLSuite/RegressionBenchmark.h"
#include "code/MLSuite/ClassificationBenchmark.h"
#include "code/MLSuite/Dataset.h"

int main() {
    try {
        // 1. Create a factory for a regression task
        ClassicModelFactory regressionFactory(
            "../data-preprocessing/data-files/regression/housing_data/housing_X_train_processed.csv",
            "../data-preprocessing/data-files/regression/housing_data/housing_y_train.csv",
            "../data-preprocessing/data-files/regression/housing_data/housing_X_test_processed.csv",
            "../data-preprocessing/data-files/regression/housing_data/housing_y_test.csv");

        // 2. Load data
        Dataset x_train = regressionFactory.loadTrainFeatures();
        Dataset y_train = regressionFactory.loadTrainTargets();
        Dataset x_test = regressionFactory.loadTestFeatures();
        Dataset y_test = regressionFactory.loadTestTargets();

        // 3. Create a Linear Regression model
        std::unique_ptr<IModel> model = regressionFactory.createLinRegModel();

        // 4. Benchmark the model
        RegressionBenchmark regressionBenchmark;
        regressionBenchmark.trainAndExecute(*model, x_train, y_train, x_test, y_test);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

## Contents 

- **data-preprocessing**: This folder contains everything related to the datasets used in the project. It is divided into three subdirectories:
  - `data-files`: Includes the pre-processed datasets for both classification and regression tasks.
  - `preprocessing-code`: Contains the Jupyter notebooks used to pre-process the raw data.
  - `source-files`: Holds the raw data files before any processing.

- **tests**: Contains all the unit tests for the project.

## Classes Definition

### `IModel`

An interface class that defines the common structure for all machine learning models. It ensures that every model implements the `fit` and `predict` methods, allowing for consistent training and evaluation across different model types, especially for internal use by the `BenchmarkStrategy`.

#### Public Methods

- **`virtual void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values) = 0;`**

  A pure virtual method for training the model.

- **`virtual std::vector<float> predict(const std::vector<float>& x_values, const std::vector<std::string>& columns) const = 0;`**

  A pure virtual method for making predictions with the trained model.

- **`virtual std::string getName() const = 0;`**

  A pure virtual method to get the name of the model.

### `BenchmarkStrategy`

An interface for the Strategy Design Pattern, allowing different benchmarking methods to be used interchangeably.

#### Public Methods

- **`virtual void execute(const IModel& model, const Dataset& testFeatures, const Dataset& testTargets) const = 0;`**

  A pure virtual method to execute the benchmark on a given model and dataset.

### `RegressionBenchmark`

A concrete implementation of `BenchmarkStrategy` for evaluating regression models.

#### Public Methods

- **`void execute(const IModel& model, const Dataset& testFeatures, const Dataset& testTargets) const override;`**

  Calculates and prints regression metrics such as Mean Squared Error (MSE) and R-squared.

### `ClassificationBenchmark`

A concrete implementation of `BenchmarkStrategy` for evaluating classification models.

#### Public Methods

- **`BenchmarkResult execute(const IModel& model, const Dataset& xData, const Dataset& yData, double fitMillis) const override;`**

  Calculates and prints classification metrics such as Accuracy, Precision, Recall and F1-score.

### `Dataset`

A class for loading and handling datasets from CSV files or in-memory data.

#### Public Methods

- **`Dataset(std::string path, std::string data_type);`**

  Constructor to load a dataset from a CSV file.

- **`Dataset(const std::vector<std::vector<float>>& features, const std::vector<float>& targets);`**

  Constructor for in-memory data.

- **`const std::vector<float>& get_data() const;`**

  Returns the raw data.

- **`const std::vector<std::vector<float>>& getFeatures() const;`**

  Returns the features for benchmarking.

- **`const std::vector<float>& getTargets() const;`**

  Returns the targets for benchmarking.

### `ClassicModelFactory`

A factory class for creating instances of different machine learning models that conform to the `IModel` interface.

#### Public Methods

- **`std::unique_ptr<IModel> createLinRegModel();`**

  Creates a linear regression model.

- **`std::unique_ptr<IModel> createLogRegModel();`**

  Creates a logistic regression model.

- **`std::unique_ptr<IModel> createRandomForestModel(int nEstimators, int maxDepth, int minSamplesSplit);`**

  Creates a random forest model with specified hyperparameters.

- **`std::unique_ptr<IModel> createXGBoostModel(int nEstimators, float learningRate, int maxDepth, ...);`**

  Creates an XGBoost model with specified hyperparameters.

### `DecisionTree`

A foundational class that implements a single decision tree. This class is used by `RandomForest` and `XGBoostModel` but does not inherit from `IModel`.

#### Public Methods

- **`DecisionTree(int maxDepth, int minSampleSplit = 2);`**

  Constructor to create a decision tree with a specified maximum depth and minimum samples for a split.

- **`void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);`**

  Trains the decision tree on the given dataset.

- **`double predict(const std::vector<double>& x) const;`**

  Makes a prediction for a single data point.

### `LinRegModel`

An implementation of a linear regression model that inherits from `IModel`.

#### Public Methods

- **`void fit(Dataset& X_dataset, Dataset& y_dataset, const std::string& regularization, double lambda);`**

  Trains the linear regression model with optional regularization.

- **`Eigen::VectorXf predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test);`**

  Makes predictions on a test set.

### `LogRegModel`

An implementation of a logistic regression model that inherits from `IModel`.

#### Public Methods

- **`void fit(const std::vector<float>& x_values, const std::vector<std::string>& columns, const std::vector<float>& y_values_vec, const std::string& regularization, double lambda, double learning_rate, int num_iterations);`**

  Trains the logistic regression model with optional regularization.

- **`Eigen::VectorXf predict(const Eigen::Ref<const Eigen::MatrixXf>& X_test) const;`**

  Makes predictions on a test set, returning class labels (0 or 1).

- **`Eigen::VectorXf predict_proba(const Eigen::Ref<const Eigen::MatrixXf>& X_test) const;`**

  Makes predictions on a test set, returning probabilities.

### `RandomForest`

An implementation of the random forest algorithm, which is an ensemble of decision trees. Inherits from `IModel`.

#### Public Methods

- **`RandomForest(int Estimators, int maxDepth, ...);`**

  Constructor to create a random forest with specified hyperparameters.

- **`void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);`**

  Trains the random forest model.

- **`double predict(const std::vector<double>& X) const;`**

  Makes a prediction for a single data point.

### `XGBoostModel`

An implementation of the XGBoost algorithm. Inherits from `IModel`.

#### Public Methods

- **`XGBoostModel(int nEstimators, float learningRate, ...);`**

  Constructor to create an XGBoost model with specified hyperparameters.

- **`void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);`**

  Trains the XGBoost model.

- **`double predict(const std::vector<double>& input) const;`**

  Makes a prediction for a single data point.

### `LinearRegressionBuilder`

A builder class for constructing `LinRegModel` instances.

#### Public Methods

- **`LinearRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);`**

  Sets the training data for the model.

- **`LinearRegressionBuilder& with_regularization(const std::string& type);`**

  Sets the regularization type (e.g., "L1" or "L2").

- **`LinearRegressionBuilder& with_lambda(double lambda);`**

  Sets the regularization strength.

- **`LinRegModel fit();`**

  Builds and trains the model.

- **`std::unique_ptr<LinRegModel> build_unfitted();`**

  Builds an unfitted model.

### `LogisticRegressionBuilder`

A builder class for constructing `LogRegModel` instances.

#### Public Methods

- **`LogisticRegressionBuilder& with_training_data(Dataset& X_train, Dataset& y_train);`**

  Sets the training data for the model.

- **`LogisticRegressionBuilder& with_regularization(const std::string& type);`**

  Sets the regularization type (e.g., "L2").

- **`LogisticRegressionBuilder& with_lambda(double lambda);`**

  Sets the regularization strength.

- **`LogisticRegressionBuilder& with_learning_rate(double rate);`**

  Sets the learning rate for gradient descent.

- **`LogisticRegressionBuilder& with_num_iterations(int iterations);`**

  Sets the number of iterations for gradient descent.

- **`LogRegModel fit();`**

  Builds and trains the model.

- **`std::unique_ptr<LogRegModel> build_unfitted();`**

  Builds an unfitted model.

### `RandomForestBuilder`

A builder class for constructing `RandomForest` instances.

#### Public Methods

- **`RandomForestBuilder& setEstimators(int estimators);`**

  Sets the number of trees in the forest.

- **`RandomForestBuilder& setMaxDepth(int maxDepth);`**

  Sets the maximum depth of the trees.

- **`std::unique_ptr<RandomForest> build();`**

  Builds the `RandomForest` model.

### `XGBoostBuilder`

A builder class for constructing `XGBoostModel` instances.

#### Public Methods

- **`XGBoostBuilder& setNEstimators(int count);`**

  Sets the number of boosting rounds.

- **`XGBoostBuilder& setLearningRate(float rate);`**

  Sets the learning rate.

- **`std::unique_ptr<XGBoostModel> build();`**

  Builds the `XGBoostModel`.


