#include "BenchmarkStrategy.h"
#include <chrono>
#include <cstddef>
#include <iostream>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#endif

// train and print time taken to train the model instead of using a timeFit function to benchmark time in main.cpp. 
BenchmarkResult BenchmarkStrategy::trainAndExecute(IModel& model, const Dataset& trainFeatures, const Dataset& trainTargets, const Dataset& testFeatures, 
						   const Dataset& testTargets) const {

	std::cout << "\n--- Benchmarking " << model.getName() << " ---" << std::endl;
    	const auto start = std::chrono::high_resolution_clock::now();
    	model.fit(trainFeatures.get_data(), trainFeatures.get_columns(), trainTargets.get_data());
    	const auto end = std::chrono::high_resolution_clock::now();
    
    	double fitMillis = millisBetween(start, end);
    	return execute(model, testFeatures, testTargets, fitMillis);
}

double currentMemoryUsageBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize);
    }
    return 0.0;
#else
    // portable placeholder for non-Windows; can be extended with getrusage or /proc
    return 0.0;
#endif
}

// return the time spent training 
double millisBetween(const std::chrono::high_resolution_clock::time_point& start, const std::chrono::high_resolution_clock::time_point& end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}
