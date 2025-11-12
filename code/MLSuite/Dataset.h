#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>

class Dataset {
private:
	// For file-based data
	std::string file_path;
	std::string type;
	std::vector<float> data;
	std::vector<std::string> columns;




public: 
	// constructor for loading dataset from a file 
	Dataset(std::string path, std::string data_type);

	
	// getters 
	const std::vector<float>& get_data() const;
	std::string get_path() const;
	std::string get_type() const;
	const std::vector<std::string>& get_columns() const;

	// helper method for reading csv 
	void read_csv(std::string path);

	// setters 
	void set_path(std::string new_path); 
	void set_data(std::vector<float> new_data, std::vector<std::string> new_cols);
	void set_type(std::string new_type);
};

#endif
