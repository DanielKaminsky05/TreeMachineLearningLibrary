#pragma once

#include <functional>
#include <string>

class DemoRunner {
public:
	using LogFn = std::function<void(const std::string&)>;

    	// Runs the demo and returns 0 on success, 1 on error.
    	static int runFullDemo(const LogFn& log);
};
