#ifndef LOGGER_H_
#define LOGGER_H_

#include <expected>
#include <format>
#include <fstream>
#include <mutex>
#include <source_location>
#include <string_view>

#include "core/config.h"

namespace deeprlzero {

class Logger {
 public:
  // Error types
  enum class Error { FileOpenError, WriteError };

  static Logger& GetInstance(const Config& config = Config()) {
    static Logger instance(config);
    return instance;
  }

  // Modern logging with std::expected and source_location
  [[nodiscard]] std::expected<void, Error> Log(
      std::string_view message,
      const std::source_location location = std::source_location::current()) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
      auto formatted_message = std::format("[{}:{}] {}", location.file_name(),
                                           location.line(), message);

      if (log_file_.is_open()) {
        log_file_ << formatted_message << std::endl;
        if (!log_file_) {
          return std::unexpected(Error::WriteError);
        }
        log_file_.flush();  // Ensure immediate write to file
      } else {
        return std::unexpected(Error::FileOpenError);
      }
      return {};
    } catch (...) {
      return std::unexpected(Error::WriteError);
    }
  }

  // Template for logging with formatting
  template <typename... Args>
  [[nodiscard]] std::expected<void, Error> LogFormat(
      std::format_string<Args...> fmt, Args&&... args) {
    try {
      auto formatted = std::format(fmt, std::forward<Args>(args)...);
      return Log(formatted);
    } catch (...) {
      return std::unexpected(Error::WriteError);
    }
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

 private:
  explicit Logger(const Config& config) {
    log_file_.open(config.log_file_path, std::ios::out | std::ios::app);
  }

  ~Logger() {
    if (log_file_.is_open()) {
      log_file_.close();
    }
  }

  std::ofstream log_file_;
  std::mutex mutex_;
};

}  // namespace deeprlzero

#endif