#include "core/logger.h"

namespace alphazero {

Logger::Logger(const Config& config)
    : log_file_(config.log_file_path) {}

void Logger::Log(const std::string& message) {
    log_file_ << message << std::endl;
}

}