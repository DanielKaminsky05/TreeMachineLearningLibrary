#include "Dataset.h"
#include <iostream>

// Use specific using declarations instead of `using namespace std;`
using std::string;
using std::vector;
using std::cerr;
using std::endl;

// new using declarations for reading files
using std::ifstream;
using std::stringstream;



/* NOTE: Dataset strategy: store a contiguous 1D vector in memory for fast access, separate metadata from the data. 
 * The Dataset class stores data in row-major order, where we unroll each row in the 1D vector.
 * Getting Data from the contiguous block: Given column count C, we can map (i, j) back to the 1D index by (i x C) + j.
 * Accessing an entire row: start index = i x C, end index = (i + 1) x C
 * Since the data is already processed, row access is ideal. 
 *
 * NOTE: heterogeneous columnar storage and block access by data type via a block manager is not considered, since neural networks are the core of the library.
 * For better open source integration, we can re-implement the data loading logic in the future with dataframe libaries and custom definitions.
 * */

/**
 *
 * @brief the Dataset class for loading the already processed datasets as a homogeneous float 1D vector and a column vector in memory.
 *
 * @param path 
 * @param data_type
 *
 * @return the Dataset object to access data from directly.
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

void Dataset::read_csv(string path) {


}

vector<float> Dataset::get_data() { // getter method for the data 
    return data;
}

string Dataset::get_path() { // getter for file path 
	return file_path;
}

string Dataset::get_type() { 
	return type; 
}


void Dataset::set_data(vector<float> new_data, vector<string> new_cols) { // getter method for the data 
	data = new_data;
	columns = new_cols;
}

void Dataset::set_path(string new_path) { 
	file_path = new_path;
}

void Dataset::set_type(string new_type) { 
	type = new_type;	
}

