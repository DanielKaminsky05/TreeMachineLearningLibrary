#ifndef CLASSICMODELFACTORY_H
#define CLASSICMODELFACTORY_H

#include "Dataset.h"
#include "HyperparameterSearch.h"
#include "IModel.h"
#include "BenchmarkStrategy.h"
#include <memory>

class ClassicModelFactory : public HyperparameterSearch {
public:
	ClassicModelFactory() = default;
	ClassicModelFactory(const std::string& trainFeaturesPath,
		const std::string& trainTargetsPath,
		const std::string& testFeaturesPath,
		const std::string& testTargetsPath,
		const std::string& trainType = "train",
		const std::string& testType = "test");

	void setTrainDataPaths(const std::string& featuresPath,
		const std::string& targetsPath,
		const std::string& type = "train");
	void setTestDataPaths(const std::string& featuresPath,
		const std::string& targetsPath,
		const std::string& type = "test");

	const std::string& getTrainFeaturesPath() const;
	const std::string& getTrainTargetsPath() const;
	const std::string& getTrainType() const;
	const std::string& getTestFeaturesPath() const;
	const std::string& getTestTargetsPath() const;
	const std::string& getTestType() const;

	Dataset loadTrainFeatures() const;
	Dataset loadTrainTargets() const;
	Dataset loadTestFeatures() const;
	Dataset loadTestTargets() const;

	void fitModel(IModel& model) const;

	std::unique_ptr<IModel> randomSearch(
		const std::string& modelType,
        const std::vector<std::vector<std::string>>& hyperParams,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const BenchmarkStrategy& evaluationStrategy,
        const LogFn& log) override;

	std::unique_ptr<IModel> createLinRegModel(); // linreg 
	std::unique_ptr<IModel> createLogRegModel();

    std::unique_ptr<IModel> createRandomForestModel(int nEstimators, int maxDepth, int minSamplesSplit, bool isClassification = false);

	std::unique_ptr<IModel> createXGBoostModel(int nEstimators = 100, float learningRate = 0.1f, int maxDepth = 3, float subsampleRatio = 1.0f, float gamma = 0.0f, const std::string& regularization = "L2", bool isClassification = false); // XGBoost 

private:
	std::string m_trainFeaturesPath;
	std::string m_trainTargetsPath;
	std::string m_testFeaturesPath;
	std::string m_testTargetsPath;
	std::string m_trainType{"train"};
	std::string m_testType{"test"};
};

#endif
