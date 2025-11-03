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
	Dataset(std::string path, std::string data_type); // constructor 
	
	// getters 
	std::vector<float> get_data();
	std::string get_path();
	std::string get_type();

	// helper method for reading csv 
	void read_csv(std::string path);

	// setters 
	void set_path(std::string new_path); 
	void set_data(std::vector<float> new_data, std::vector<std::string> new_cols);
	void set_type(std::string new_type);
};

#endif
