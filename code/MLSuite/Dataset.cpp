#include <iostream>
#include <vector>
#include <string>

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
class Dataset {
// private variables 
private: 
	std::string file_path;
	std::string type; 
	
	std::vector<float> data; // TODO: casting everything to a float
	std::vector<std::string> columns;

// constructor 
	Dataset (string path, string data_type) : file_path(path), type(data_type) {
		// if the data_type is not "train", "test", or "val", throw an exception.
		// using the string path, read the csv 
	}


// public methods 
public: 
	std::vector<float> get_data() { // getter method for the data 
		return data;
	}

	std::string get_path() { return file_path; }
	std::string get_type() { return type; }

	void read_csv(string path) {
		return;

	}
}


