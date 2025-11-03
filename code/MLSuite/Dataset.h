#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

class Dataset {
private:
	std::string file_path;
	std::string type;
	std::vector<float> data;
	std::vector<std::string> columns;

public: 
	Dataset(std::string path, std::string data_type);
	
	std::vector<float> get_data();
	std::string get_path();
	std::string get_type();

void read_csv(std::string path);
};

#endif
