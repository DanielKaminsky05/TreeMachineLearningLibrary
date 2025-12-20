#include "Dataset.h"
#include <iostream>
#include <stdexcept>

// Use specific using declarations instead of `using namespace std;
using std::string;
using std::vector;
using std::cerr;
using std::endl;

// using declarations for reading files
using std::ifstream;
using std::stringstream;
using std::to_string;


/* NOTE: Dataset strategy: store a contiguous 1D vector in memory for fast access, separate metadata from the data. 
 * The Dataset class stores data in row-major order, where we unroll each row in the 1D vector.
 * Getting Data from the contiguous block: Given column count C, we can map (i, j) back to the 1D index by (i x C) + j.
 * Accessing an entire row: start index = i x C, end index = (i + 1) x C
 * Since the data is already processed, row access is ideal. 
 *
 * NOTE: heterogeneous columnar storage and block access by data type via a block manager is not considered, since neural networks are the core of the library.
 * For better open source integration, we can re-implement the data loading logic in the future with dataframe libaries and custom definitions.
 * */

Dataset::Dataset(string path, string data_type) : file_path(path), type(data_type) {
    	// if the data_type is not "train", "test", or "val", throw an exception.
	if (data_type != "train" && data_type != "test" && data_type != "val") {
		throw std::invalid_argument("Invalid dataset type, must be train, test or val:" + data_type);
	}

    	// using the string path, read the csv 
	try {
		read_csv(path);	
	} catch (const std::runtime_error& e) {
		cerr << "Error reading file: " << e.what() << endl; 
	} catch (...) {
		cerr << "Unknown error" << endl;
	}
}

Dataset::Dataset(const std::vector<std::vector<float>>& features, const std::vector<float>& targets) {
    type = "in-memory";
    data.clear();
    columns.clear();

    if (!features.empty()) {
        // assume this is a feature dataset
        size_t n_rows = features.size();
        size_t n_cols = features[0].size();

        // generate dummy column names
        for (size_t j = 0; j < n_cols; ++j) {
            columns.push_back(to_string(j));
        }

        // Flatten data
        data.reserve(n_rows * n_cols);
        for (const auto& row : features) {
            if (row.size() != n_cols) {
                throw std::invalid_argument("Inconsistent feature row sizes");
            }
            data.insert(data.end(), row.begin(), row.end());
        }
    } else if (!targets.empty()) {
        columns.push_back("target");
        data = targets;
    }
}

void Dataset::read_csv(string path) {
	// open file, throw exception if it is not
	ifstream file(path);

    	if (!file.is_open()) {
        	throw std::runtime_error("Error: file not found at " + path);
    	}

    	// clear existing data
    	columns.clear();
    	data.clear();

    	// get the first line, which is the columns.
    	string col_line;
    	if (getline(file, col_line)) {
        	stringstream ss(col_line);
        	string cell;

        	while (getline(ss, cell, ',')) {
            		columns.push_back(cell);
        	}
    }

    	// get the values and put them into the data 1D vector.
    	string line;
    	while (getline(file, line)) {
        	stringstream ss(line);
        	string cell;

        while (getline(ss, cell, ',')) {
        	try {
                	data.push_back(std::stof(cell));
            	} catch (const std::invalid_argument& e) {
                	cerr << "Could not convert string to float: " << cell << endl;
            	}
        }
    }
}

const vector<float>& Dataset::get_data() const { // getter method for the data 
	return data;
}

string Dataset::get_path() const { // getter for file path 
	return file_path;
}

string Dataset::get_type() const { 
	return type; 
}

const vector<string>& Dataset::get_columns() const {
	return columns;
}

// helper conversion functions 
std::vector<std::vector<double>> Dataset::get_data_as_double_2d() const {
    size_t n_cols = columns.size();
    if (n_cols == 0) return {};
    size_t n_rows = data.size() / n_cols;
    
    std::vector<std::vector<double>> out(n_rows, std::vector<double>(n_cols));
    for(size_t i=0; i<n_rows; ++i) {
        for(size_t j=0; j<n_cols; ++j) {
            out[i][j] = static_cast<double>(data[i*n_cols + j]);
        }
    }
    return out;
}

std::vector<double> Dataset::get_data_as_double_1d() const {
    std::vector<double> out(data.size());
    for(size_t i=0; i<data.size(); ++i) {
        out[i] = static_cast<double>(data[i]);
    }
    return out;
}

// setters for Dataset class, used in classic model factory
void Dataset::set_data(vector<float> new_data, vector<string> new_cols) { 
	data = new_data;
	columns = new_cols;
}

void Dataset::set_path(string new_path) { 
	file_path = new_path;
}

void Dataset::set_type(string new_type) { 
	type = new_type;	
}
