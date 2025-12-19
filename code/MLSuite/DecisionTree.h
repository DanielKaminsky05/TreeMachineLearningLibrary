#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include <tuple>
class DecisionTree
{
private:
	int maxDepth;
	int minSampleSplit;
    	bool isClassification;
	int nNodes;
    	int nFeatures;
    	bool isFitted;
    	std::vector<int> feature;
    	std::vector<double> threshold;
    	std::vector<int> left;
    	std::vector<int> right;
    	std::vector<bool> isLeaf;
    	std::vector<double> value;
    	double sumY = 0.0;
    	std::vector<double> sumY2;
    	void buildTree(const std::vector<std::vector<double>>& X,const std::vector<double>& Y,const std::vector<int>& indices, int depth, int nodeIndex);
    	std::tuple<int, double, double, std::vector<int>, std::vector<int>> bestSplit(const std::vector<std::vector<double>>& X,const std::vector<double>& Y,const std::vector<int>& indices);
    	double computeMSE(int n, double sum, double sum2);
        double computeGini(const std::vector<int>& indices, const std::vector<double>& Y);
    	double impurityDecrease(int nP, double sumP, double sumP2, int nL, double sumL, double sumL2, int nR, double sumR, double sumR2,
                              const std::vector<int>& indicesP, const std::vector<int>& indicesL, const std::vector<int>& indicesR, const std::vector<double>& Y);
    	void makeLeaf(int nodeIndex,const std::vector<int>& indicies, const std::vector<double>& Y);
    	std::tuple<std::vector<int>, std::vector<int>> partitionByThreshold(const std::vector<std::vector<double>>& X, int feat, double thr,const std::vector<int>& indicies);
    	int newNode();

public:
    	DecisionTree(int maxDepth, int minSampleSplit = 2, bool isClassification = false);
    	void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    	double predict(const std::vector<double>& x) const;
    	int getNNodes() const { return nNodes; }
};

#endif 
