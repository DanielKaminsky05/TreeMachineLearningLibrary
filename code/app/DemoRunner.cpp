#include "code/app/DemoRunner.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "code/MLSuite/BenchmarkStrategy.h"
#include "code/MLSuite/ClassicModelFactory.h"
#include "code/MLSuite/ClassificationBenchmark.h"
#include "code/MLSuite/Dataset.h"
#include "code/MLSuite/IModel.h"
#include "code/MLSuite/RegressionBenchmark.h"

namespace {
void logLine(const DemoRunner::LogFn& log, const std::string& line) {
    if (log) {
        log(line);
    } else {
        std::cout << line << std::endl;
    }
}

std::string formatFloat(float value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
}
} // namespace

int DemoRunner::runFullDemo(const LogFn& log) {
    try {
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

        {
            logLine(log, "--- Benchmarking Linear Regression ---");
            std::unique_ptr<IModel> model = regressionFactory.createLinRegModel();
            double fitMs = timeFit(*model, x_train, y_train);
            benchmark.execute(*model, x_test, y_test, fitMs);

            std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
            for (int i = 0; i < 10 && i < static_cast<int>(results.size()); ++i) {
                logLine(log, "Pred[" + std::to_string(i) + "] = " + formatFloat(results[i]));
            }
        }

        {
            logLine(log, "--- Benchmarking Random Forest ---");
            std::unique_ptr<IModel> model = regressionFactory.createRandomForestModel(50, 10, 2);
            double fitMs = timeFit(*model, x_train, y_train);
            benchmark.execute(*model, x_test, y_test, fitMs);

            std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
            for (int i = 0; i < 10 && i < static_cast<int>(results.size()); ++i) {
                logLine(log, "Pred[" + std::to_string(i) + "] = " + formatFloat(results[i]));
            }
        }

        {
            logLine(log, "--- Benchmarking XGBoost ---");
            std::unique_ptr<IModel> model =
                regressionFactory.createXGBoostModel(50, 0.1f, 10, 0.8f, 0.1f, "L2");
            double fitMs = timeFit(*model, x_train, y_train);
            benchmark.execute(*model, x_test, y_test, fitMs);

            std::vector<float> results = model->predict(x_test.get_data(), x_test.get_columns());
            for (int i = 0; i < 10 && i < static_cast<int>(results.size()); ++i) {
                logLine(log, "Pred[" + std::to_string(i) + "] = " + formatFloat(results[i]));
            }
        }

        {
            logLine(log, "--- Classification Benchmark: Random Forest (Iris) ---");
            std::unique_ptr<IModel> model = classificationFactory.createRandomForestModel(50, 10, 2);
            double fitMs = timeFit(*model, cx_train, cy_train);
            classificationBenchmark.execute(*model, cx_test, cy_test, fitMs);
        }

        logLine(log, "--- Demo Complete ---");
        return 0;
    } catch (const std::exception& e) {
        logLine(log, std::string("An error occurred: ") + e.what());
        return 1;
    }
}
