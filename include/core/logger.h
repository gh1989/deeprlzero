#ifndef ALPHAZERO_LOGGER_H_
#define ALPHAZERO_LOGGER_H_

#include <iostream>
#include <string>
#include <fstream>
#include "core/config.h"

namespace alphazero {

class Logger {
public:
    Logger(const Config& config);
    void Log(const std::string& message);
private:
    std::ofstream log_file_;
    Config config_;
};

}

#endif