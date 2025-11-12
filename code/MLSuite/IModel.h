#ifndef IMODEL_H
#define IMODEL_H

#include <vector>
#include <string>

// IModel is a common interface for the benchmark class to use and ensure consistent behavior across all types of models
// so we do not have to modify benchmark for each model type with the IModel interface, as long as the model can use fit() and predict().
class IModel {
public:
	virtual ~IModel() = default;

    	// Method for fitting the model to training data
    	virtual void fit(const std::vector<float>& x_values,
	const std::vector<std::string>& columns,
        const std::vector<float>& y_values) = 0;

    	virtual std::vector<float> predict(const std::vector<float>& x_values,
        const std::vector<std::string>& columns) const = 0; // return predictions

    	// Method to get the name of the model (used by benchmark)
    	virtual std::string getName() const = 0;
};

#endif 
