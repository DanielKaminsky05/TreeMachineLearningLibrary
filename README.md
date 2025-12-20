# 3307 Final OOP Deliverable

This README details the final submission for our course project for OOP design patterns, where we implemented traditional machine learning libraries in C++ from scratch with our custom Dataset implementation. Users can load CSV data, train the model, and benchmark the results and performance of the model using our abstracted library.

[Demonstration Video Link](https://drive.google.com/file/d/1hTJA2yeSf_bWunvn1wNWdFWe47EQi0Iw/view?usp=sharing)

For more implementation details or details about building and running the code, class API definitions and details about Datasets used for this project, please refer to the [report README](report/README.md) for inside the report folder, which also contains the Documentation PDFs submitted for Deliverable 3, for further instructions.

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
├── report # report for deliverables, and documentation and instructions for building and running in the README.md file.
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

## Class definition and API usage

Please refer to the [report README](report/README.md) and the [data README](data-preprocessing/README.md) for more details about the data, class definitions, and running the code.

We used generative AI tools such as Gemini and ChatGPT to brainstorm, give feedback and validate our use of design patterns for our project code. 
