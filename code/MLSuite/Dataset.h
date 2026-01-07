#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>

class Dataset {
private:

	std::string file_path;
	std::string type;
	std::vector<float> data;
	std::vector<std::string> columns;

public: 
	// constructor for loading dataset from a file 
	Dataset(std::string path, std::string data_type);

    // constructor for creating dataset from in-memory vectors
    Dataset(const std::vector<std::vector<float>>& features, const std::vector<float>& targets);
	
	// getters 
	const std::vector<float>& get_data() const;
	std::string get_path() const;
	std::string get_type() const;
	const std::vector<std::string>& get_columns() const;

	// helper method for reading csv 
	void read_csv(std::string path);

    // Helpers to export data as double for specific use cases (like RandomSearch)
    std::vector<std::vector<double>> get_data_as_double_2d() const;
    std::vector<double> get_data_as_double_1d() const;

	// setters 
	void set_path(std::string new_path); 
	void set_data(std::vector<float> new_data, std::vector<std::string> new_cols);
	void set_type(std::string new_type);
};

#endif
