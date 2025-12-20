# 3307 Final OOP Deliverable

This README details the final submission for our course project for OOP design patterns, where we implemented traditional machine learning libraries in C++ from scratch with our custom Dataset implementation. Users can load CSV data, train the model, and benchmark the results and performance of the model using our abstracted library.

For more implementation details, please refer to the [report README](report/README.md).

# Repo structure 

```
.
├── 3rdparty
│   ├── eigen-5.0.0
│   └── eigen-5.0.0.zip
├── code 
│   ├── app
│   │   ├── DemoRunner.cpp
│   │   └── DemoRunner.h 
│   ├── MLSuite 
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
│   └── ui 
│       ├── MainWindow.cpp
│       └── MainWindow.h 
├── data-preprocessing
│   ├── data-files
│   ├── preprocessing-code
│   ├── source-files
│   └── README.md
├── report
│   ├── 3307 Deliverable #2 - Henrique & Daniel.pdf 
│   ├── 3307 Deliverable #2 - Thomson Lam.pdf
│   ├── 3307 Deliverable #3 - Henrique & Daniel.pdf 
│   └── 3307 Deliverable #3 - Thomson Lam.pdf
│   └── README.md
├── tests
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
├── demo.cpp
├── main.cpp
└── README.md 
```

## Class definition and API usage

Please refer to the [report README](report/README.md) and the [data README](data-preprocessing/README.md) for more details about the data, class definitions, and running the code.

We used generative AI tools such as Gemini and ChatGPT to brainstorm, give feedback and validate our use of design patterns for our project code. 
