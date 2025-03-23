#ifndef ALPHAZERO_NETWORK_MANAGER_H_
#define ALPHAZERO_NETWORK_MANAGER_H_

#include "core/neural_network.h"
#include "core/config.h"
#include "core/evaluator.h"
#include <memory>
#include <string>
#include "core/logger.h"

namespace alphazero {

class NetworkManager {
 public:
  explicit NetworkManager(const Config& config);
  
  std::shared_ptr<NeuralNetwork> CreateInitialNetwork();
  void SetBestNetwork(std::shared_ptr<NeuralNetwork> network);
  bool LoadBestNetwork();
    
  bool AcceptOrRejectNewNetwork(std::shared_ptr<NeuralNetwork> network, EvaluationStats evaluation_stats);
  void UpdateTemperature();
  float GetCurrentTemperature() const { return current_temperature_; }
  std::shared_ptr<NeuralNetwork> GetBestNetwork();
  
  void SaveBestNetwork() const;
  
 private:
  const Config& config_;
  std::shared_ptr<NeuralNetwork> best_network_;
  EvaluationStats best_evaluation_stats_;
  float current_temperature_;
  int best_iteration_ = 0;
  int current_iteration_ = 0;
  Logger& logger_ = Logger::GetInstance();
};

}  // namespace alphazero

#endif  // ALPHAZERO_NETWORK_MANAGER_H_ 