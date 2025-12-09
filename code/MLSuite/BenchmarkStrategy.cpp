#include "BenchmarkStrategy.h"
#include <chrono>
#include <cstddef>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#endif

double currentMemoryUsageBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize);
    }
    return 0.0;
#else
    // Portable placeholder for non-Windows; can be extended with getrusage or /proc.
    return 0.0;
#endif
}

double millisBetween(const std::chrono::high_resolution_clock::time_point& start,
                     const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
